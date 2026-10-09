"""CPU tests for Qwen2.5-VL support in the COCO-VTR pipeline.

  * geometry: our resize rule and token grid equal the real Qwen image processor's, and the
    token order matches the processor's pixel layout (a bright square lands on the token our
    geometry says covers it), for several image sizes
  * capture: hidden states of the image tokens equal HF hidden_states, token count = geometry
  * Patchscopes / SelfIE on Qwen: cached scoring == full-sequence scoring (exercises the
    explicit M-RoPE text positions), and injection changes the scores
  * logit / tuned / embedding lens, LatentLens bank build + search, probes, and run_vtr's
    readout + eval stages on a tiny random Qwen2.5-VL

Needs network access once for the Qwen2.5-VL tokenizer and image-processor configs (no weights).

  python test_qwen.py
"""

import json
import os
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import latentlens_object_identification as LL  # noqa: E402
import run_vtr  # noqa: E402
from general_utils import ModelAndTokenizer, QwenVLImageTextProcessor  # noqa: E402
from vtr import evaluate as E  # noqa: E402
from vtr.cache import cache_states  # noqa: E402
from vtr.data import select_object_samples  # noqa: E402
from vtr.geometry import Qwen25Geometry  # noqa: E402
from vtr.models import capture_states  # noqa: E402
from vtr.readouts import (EmbeddingLens, LogitLens, Patchscopes, TunedLens,  # noqa: E402
                          class_forms)

NAME = "Qwen/Qwen2.5-VL-7B-Instruct"


def load_processor():
  from transformers import AutoTokenizer
  from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
  tok = AutoTokenizer.from_pretrained(NAME)
  return QwenVLImageTextProcessor(Qwen2VLImageProcessor.from_pretrained(NAME), tok)


def test_geometry(processor):
  geo = Qwen25Geometry()
  ip = processor.image_processor
  mean = np.array(ip.image_mean).reshape(1, 3, 1, 1, 1)
  std = np.array(ip.image_std).reshape(1, 3, 1, 1, 1)
  for h, w in ((480, 640), (640, 480), (333, 500), (427, 640), (200, 1000)):
    rows, cols = geo.grid(h, w)
    out = ip(images=[Image.fromarray(np.zeros((h, w, 3), np.uint8))], return_tensors="pt")
    t, gh, gw = out["image_grid_thw"][0].tolist()
    assert (gh // 2, gw // 2) == (rows, cols), ((h, w), (gh, gw), (rows, cols))

    # A bright square in the middle of one token cell (in resized coordinates).
    h_bar, w_bar = geo.resized(h, w)
    r0, c0 = rows // 2, cols // 3
    img = np.zeros((h, w, 3), np.uint8)
    y0, y1 = int((r0 * 28 + 8) * h / h_bar), int((r0 * 28 + 20) * h / h_bar)
    x0, x1 = int((c0 * 28 + 8) * w / w_bar), int((c0 * 28 + 20) * w / w_bar)
    img[y0:y1, x0:x1] = 255
    pv = ip(images=[Image.fromarray(img)], return_tensors="pt")["pixel_values"].numpy()
    # pixel_values: [patches, C * temporal * 14 * 14]; 4 consecutive patches = 1 LLM token.
    raw = pv.reshape(-1, 3, 2, 14, 14) * std + mean
    brightness = raw.reshape(rows * cols, 4, -1).mean(axis=(1, 2))
    assert int(brightness.argmax()) == r0 * cols + c0, (int(brightness.argmax()), r0 * cols + c0)

    mask = np.zeros((h, w), bool)
    mask[y0:y1, x0:x1] = True
    cov = geo.coverage(mask)
    assert cov.shape == (rows * cols,) and int(cov.argmax()) == r0 * cols + c0
  print("geometry OK (grid and token order match the Qwen image processor)")


def build_tiny_qwen(processor):
  from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
  tok = processor.tokenizer
  config = Qwen2_5_VLConfig(
      text_config={"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 4,
                   "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": 151936,
                   "max_position_embeddings": 4096,
                   "rope_scaling": {"type": "mrope", "mrope_section": [2, 3, 3]}},
      vision_config={"depth": 2, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2,
                     "out_hidden_size": 64, "patch_size": 14, "spatial_merge_size": 2,
                     "temporal_patch_size": 2, "window_size": 112, "fullatt_block_indexes": [1],
                     "in_channels": 3},
      image_token_id=tok.convert_tokens_to_ids("<|image_pad|>"),
      video_token_id=tok.convert_tokens_to_ids("<|video_pad|>"),
      vision_start_token_id=tok.convert_tokens_to_ids("<|vision_start|>"),
      vision_end_token_id=tok.convert_tokens_to_ids("<|vision_end|>"))
  torch.manual_seed(0)
  model = Qwen2_5_VLForConditionalGeneration(config).eval()
  mt = ModelAndTokenizer("tiny-qwen2.5-vl", model=model, tokenizer=tok, device="cpu")
  mt.processor = processor
  return mt


def make_fake_coco(root, sizes):
  os.makedirs(os.path.join(root, "val2017"), exist_ok=True)
  os.makedirs(os.path.join(root, "annotations"), exist_ok=True)
  rng = np.random.RandomState(0)
  images, anns = [], []
  for i, (h, w) in enumerate(sizes):
    fn = f"{i:012d}.jpg"
    Image.fromarray(rng.randint(0, 255, (h, w, 3), dtype=np.uint8)).save(
        os.path.join(root, "val2017", fn))
    images.append({"id": i, "file_name": fn, "height": h, "width": w})
    x0, y0 = w // 3, h // 3
    anns.append({"id": 100 + i, "image_id": i, "category_id": 1 + (i % 3), "iscrowd": 0,
                 "area": 25000.0,
                 "segmentation": [[x0, y0, x0 + 160, y0, x0 + 160, y0 + 160, x0, y0 + 160]]})
  path = os.path.join(root, "annotations", "instances_val2017.json")
  with open(path, "w", encoding="utf-8") as f:
    json.dump({"images": images, "annotations": anns,
               "categories": [{"id": 1, "name": "dog"}, {"id": 2, "name": "cat"},
                              {"id": 3, "name": "bottle"}]}, f)
  return path


def main():
  processor = load_processor()
  test_geometry(processor)
  mt = build_tiny_qwen(processor)
  assert mt.family == "qwen2.5-vl" and mt.num_layers == 4, (mt.family, mt.num_layers)
  geo = Qwen25Geometry()

  with tempfile.TemporaryDirectory() as tmp:
    coco = os.path.join(tmp, "coco")
    ann = make_fake_coco(coco, [(480, 640), (640, 480), (427, 640), (480, 640)])
    samples, class_names = select_object_samples(ann, geo, n_images=4, per_class=4,
                                                 min_area=20000, max_area=None, min_tokens=4,
                                                 seed=0)
    assert len(samples) == 4 and all(len(s["coverage"]) == s["grid"][0] * s["grid"][1]
                                      for s in samples)
    image = Image.open(os.path.join(coco, "val2017", samples[0]["file_name"])).convert("RGB")
    full = capture_states(mt, processor, image, geo, verify=True)
    assert full.shape[:2] == (4, len(samples[0]["coverage"])), full.shape

    states, meta = cache_states(mt, processor, samples, coco, 2, 2, 0, geo)
    assert states.shape[1:] == (4, 64)
    print(f"capture + cache OK {tuple(states.shape)} {meta['kind'].value_counts().to_dict()}")

    forms = class_forms(class_names, synonyms=False)
    h = states[:3, 2].float()
    for prompt in ("identity", "selfie"):
      fast = Patchscopes(mt, class_names, forms, prompt, "cpu", rows_per_batch=8)
      slow = Patchscopes(mt, class_names, forms, prompt, "cpu", rows_per_batch=8,
                         use_cache=False)
      r1, _ = fast.scores(h, 2)
      r2, _ = slow.scores(h, 2)
      assert np.allclose(r1, r2, atol=1e-3), (prompt, np.abs(r1 - r2).max())
      assert not np.allclose(r1[0], fast.prior.numpy()[:len(class_names)], atol=1e-5)
      if prompt == "selfie":
        assert len(fast.pos) == 5
    print("patchscopes + selfie OK on Qwen (cached == full; M-RoPE text positions)")

    s = LogitLens(mt, class_names, forms, "cpu").scores(h)
    assert s.shape == (3, 3) and np.isfinite(s).all()
    assert EmbeddingLens(mt, class_names, forms, "cpu").scores(h).shape == (3, 3)
    zero = os.path.join(tmp, "zero.pt")
    torch.save({"A": torch.zeros(4, 64, 64), "b": torch.zeros(4, 64)}, zero)
    assert np.allclose(TunedLens(mt, class_names, forms, "cpu", zero).scores(h, 2), s, atol=1e-4)

    corpus = ["A small dog ran across the park.", "The cat slept on the warm windowsill.",
              "She opened a bottle of water.", "Two dogs and a cat played outside."]
    bank = LL.build_bank(mt.model.model.language_model, mt.tokenizer, corpus, [1, 2, 3],
                         batch_size=2)
    bank_dir = os.path.join(tmp, "bank")
    bank.save(bank_dir)
    print(f"lenses + LatentLens bank OK ({bank})")

    out_dir = os.path.join(tmp, "out")
    os.makedirs(out_dir)
    with open(os.path.join(out_dir, "samples.json"), "w", encoding="utf-8") as f:
      json.dump({"class_names": class_names, "geometry": geo.name, "samples": samples}, f)
    torch.save({"states": states, "meta": meta}, os.path.join(out_dir, "states.pt"))
    args = run_vtr.parse_args(["--out_dir", out_dir, "--device", "cpu", "--model_name",
                               "Qwen/Qwen2.5-VL-7B-Instruct", "--latentlens_bank", bank_dir,
                               "--tuned_lens", zero, "--n_boot", "20",
                               "--readouts", "logit_lens", "tuned_lens", "embedding_lens",
                               "latentlens", "patchscopes", "selfie", "probe",
                               "--ps_rows_per_batch", "16", "--probe_splits", "2"])
    run_vtr.run_readouts(args, mt, states, meta, class_names, [0, 1, 2, 3])
    run_vtr.main(["--out_dir", out_dir, "--stage", "eval", "--n_boot", "20",
                  "--model_name", "Qwen/Qwen2.5-VL-7B-Instruct"])
    for f in ("headline.csv", "fair_classes.json"):
      assert os.path.exists(os.path.join(out_dir, f)), f
    print("run_vtr readouts + eval OK on Qwen")
  print("\nALL QWEN TESTS PASSED")


if __name__ == "__main__":
  main()
