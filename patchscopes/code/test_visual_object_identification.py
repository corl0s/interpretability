"""CPU smoke test for visual_object_identification.py.

Runs the whole pipeline on a tiny randomly-initialised LLaVA and 4 synthetic COCO images,
so the geometry, hooks, batching and scoring can be checked without a GPU or the 7B
checkpoint. Accuracies are meaningless here (random weights); what is tested is that the
machinery runs and that patching actually modifies the forward pass.

Needs network access once, to fetch the LLaVA processor/tokenizer config (no weights).

  python test_visual_object_identification.py
"""

import json
import os
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import visual_object_identification as V  # noqa: E402
from general_utils import ModelAndTokenizer, make_inputs  # noqa: E402
from patchscopes_utils import remove_hooks, set_hs_patch_hooks_llava_batch  # noqa: E402


def test_geometry():
  """Patch coverage must track where the mask actually is after resize + centre crop."""
  mask = np.zeros((480, 640), dtype=bool)
  mask[200:280, 300:380] = True
  cov = V.patch_coverage(mask)
  assert cov.shape == (24, 24)
  assert 9 <= (cov >= 0.5).sum() <= 25, (cov >= 0.5).sum()
  assert (V.patch_coverage(np.ones((480, 640), bool)) == 1).all()
  assert (V.patch_coverage(np.zeros((480, 640), bool)) == 0).all()
  top_left = np.zeros((480, 640), bool)
  top_left[0:100, 0:100] = True
  cov = V.patch_coverage(top_left)
  assert cov[:6, :6].max() > 0.5 and cov[12:, 12:].max() == 0
  print("geometry OK")


def test_scoring():
  assert V.mentions("it is a dog running", "dog")
  assert V.mentions("two dogs", "dog")
  assert not V.mentions("dogma", "dog")
  assert not V.mentions("hot dog stand", "cat")
  assert V.first_segment("dog\nfoo -> foo", "identity") == "dog"
  assert V.first_segment(": a dog, Samsung: x", "entity") == ": a dog"
  print("scoring OK")


def make_fake_coco(root):
  """4 noise images, each with one 160x160 target object and one tiny distractor."""
  os.makedirs(os.path.join(root, "val2017"), exist_ok=True)
  os.makedirs(os.path.join(root, "annotations"), exist_ok=True)
  rng = np.random.RandomState(0)
  images, annotations = [], []
  for i in range(4):
    file_name = f"{i:012d}.jpg"
    Image.fromarray(rng.randint(0, 255, (480, 640, 3), dtype=np.uint8)).save(
        os.path.join(root, "val2017", file_name))
    images.append({"id": i, "file_name": file_name, "height": 480, "width": 640})
    x0, y0 = 250 + 10 * i, 180
    annotations.append({
        "id": 100 + i, "image_id": i, "category_id": 1 + (i % 2), "iscrowd": 0,
        "area": 25000.0,
        "segmentation": [[x0, y0, x0 + 160, y0, x0 + 160, y0 + 160, x0, y0 + 160]]})
    annotations.append({
        "id": 200 + i, "image_id": i, "category_id": 3, "iscrowd": 0, "area": 900.0,
        "segmentation": [[10, 10, 40, 10, 40, 40, 10, 40]]})
  ann_path = os.path.join(root, "annotations", "instances_val2017.json")
  with open(ann_path, "w", encoding="utf-8") as f:
    json.dump({"images": images, "annotations": annotations,
               "categories": [{"id": 1, "name": "dog"}, {"id": 2, "name": "cat"},
                              {"id": 3, "name": "bottle"}]}, f)
  return ann_path


def build_tiny_vlm():
  from transformers import (CLIPVisionConfig, LlamaConfig, LlavaConfig,
                            LlavaForConditionalGeneration, LlavaProcessor)
  processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
  config = LlavaConfig(
      vision_config=CLIPVisionConfig(hidden_size=32, intermediate_size=64,
                                     num_hidden_layers=2, num_attention_heads=2,
                                     image_size=336, patch_size=14),
      text_config=LlamaConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                              num_attention_heads=4, num_key_value_heads=4,
                              vocab_size=32064),
      image_token_index=32000, vision_feature_layer=-2,
      vision_feature_select_strategy="default")
  torch.manual_seed(0)
  model = LlavaForConditionalGeneration(config).eval()
  mt = ModelAndTokenizer("llava-tiny", model=model, tokenizer=processor.tokenizer,
                         device="cpu")
  mt.processor = processor
  return mt, processor


def test_patch_injection(mt):
  """The patch hook must overwrite exactly the target position and propagate onward."""
  inp = make_inputs(mt.tokenizer, ["cat -> cat"], "cpu")
  seq_len = inp["input_ids"].shape[1]
  vec = torch.arange(64, dtype=torch.float32) * 0.01
  target_layer, position = 2, seq_len - 1
  layers = V.get_decoder(mt).layers
  captured = {}

  hooks = set_hs_patch_hooks_llava_batch(
      mt.model, [{"batch_idx": 0, "layer_target": target_layer,
                  "position_target": position, "hidden_rep": vec,
                  "skip_final_ln": False}], generation_mode=True)
  # NOTE: register the probes *after* the patch hooks -- hooks fire in registration order,
  # so a probe registered first would observe the pre-patch output.
  def probe(module, inp_, out):
    hs = out[0] if isinstance(out, tuple) else out
    captured["target"] = hs[0, position].clone()
    captured["neighbour"] = hs[0, position - 1].clone()
  handles = [layers[target_layer].register_forward_hook(probe),
             layers[target_layer + 1].register_forward_pre_hook(
                 lambda m, i: captured.__setitem__("next_input", i[0][0, position].clone()))]
  try:
    with torch.no_grad():
      mt.model(input_ids=inp["input_ids"], attention_mask=inp["attention_mask"])
  finally:
    remove_hooks(hooks)
    remove_hooks(handles)

  assert torch.allclose(captured["target"], vec), "injection did not land"
  assert torch.allclose(captured["next_input"], vec), "injection did not propagate"
  assert not torch.allclose(captured["neighbour"], vec), "wrong position overwritten"
  print("patch injection OK")


class Args:
  seed = 0
  patches_per_image = 2
  outside_per_image = 1
  batch_size = 8
  max_gen_len = 6
  prompts = ["identity", "entity"]
  probe_min_images = 2

  def __init__(self, coco_dir, out_dir):
    self.coco_dir = coco_dir
    self.out_dir = out_dir


def main():
  test_geometry()
  test_scoring()

  with tempfile.TemporaryDirectory() as tmp:
    coco_dir = os.path.join(tmp, "coco")
    out_dir = os.path.join(tmp, "out")
    os.makedirs(out_dir)
    ann_path = make_fake_coco(coco_dir)

    samples = V.select_samples(ann_path, n_images=4, min_area=20000, max_area=30000,
                               min_object_patches=4, coverage_threshold=0.5, seed=0)
    assert len(samples) == 4, samples
    assert all(len(s["object_patches"]) >= 4 for s in samples)
    assert all(s["class_name"] in ("dog", "cat") for s in samples)  # 900px one excluded
    print("sample selection OK")

    mt, processor = build_tiny_vlm()
    assert mt.is_vlm and mt.num_layers == 4
    test_patch_injection(mt)

    args = Args(coco_dir, out_dir)
    states, meta, lens = V.run_extraction(mt, processor, samples, args)
    assert states.shape == (len(meta), 4, 64)
    assert set(meta["kind"]) == {"object", "outside"}
    assert len(lens) == sum(len(s["object_patches"]) for s in samples) * 4
    print("extraction OK")

    layers = list(range(4))
    baseline = V.run_no_injection_baseline(mt, meta, args)
    rows = V.run_patchscopes(mt, states, meta, args, layers)
    assert set(rows["kind"]) == {"object", "outside", "random"}
    assert rows["generation"].notna().all()
    base_gen = dict(zip(baseline["prompt_key"], baseline["generation"].fillna("")))
    differs = sum(g != base_gen[p] for g, p in zip(rows["generation"], rows["prompt_key"]))
    assert differs > 0, "patched generations identical to baseline -- hooks not firing"
    print(f"patchscopes OK ({differs}/{len(rows)} generations differ from baseline)")

    probe = V.run_probe(states, meta, layers, args)
    table, headline = V.summarize(lens, rows, probe, args, layers)
    assert len(table) == 4 and "logit_lens_top1" in table
    print("summary OK")
    V.save_debug_overlays(processor, samples, args, n=2)

  print("\nALL TESTS PASSED")


if __name__ == "__main__":
  main()
