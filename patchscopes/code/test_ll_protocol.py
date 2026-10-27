"""CPU test for run_ll_protocol.py on a tiny random LLaVA, with their judge code and a fake judge:

  * extract / latentlens / ours write JSON their judge script parses (method auto-detection,
    sub-word expansion for LatentLens, 5 candidate words per judged patch)
  * export_lm reproduces LLaVA's own text logits
  * plan.json replays their sampling: every judged patch is in the plan, 100% coverage
  * summary reads the judge output

  python test_ll_protocol.py
"""

import json
import os
import sys
import tempfile
import types

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_ll_protocol as R  # noqa: E402
from test_visual_object_identification import build_tiny_vlm  # noqa: E402
from vtr.injection import InjectionReadout  # noqa: E402
from vtr.oracle import add_lora, lora_parameters, set_lora  # noqa: E402

LL_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "external",
                       "latentlens")


def stub_torchvision():
  """Their judge utils import torchvision (used only for Molmo's resize-and-pad)."""
  try:
    import torchvision  # noqa: F401
  except ImportError:
    tv = types.ModuleType("torchvision")
    tr = types.ModuleType("torchvision.transforms")
    fn = types.ModuleType("torchvision.transforms.functional")
    tr.InterpolationMode = types.SimpleNamespace(BILINEAR="bilinear")
    tr.Resize = None
    fn.convert_image_dtype = None
    tv.transforms, tr.functional = tr, fn
    sys.modules.update({"torchvision": tv, "torchvision.transforms": tr,
                        "torchvision.transforms.functional": fn})


class FakeOpenAI:
  """Stands in for the vLLM server: 'interpretable' iff a candidate word starts with a letter."""

  def __init__(self, **kw):
    self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

  def create(self, model, messages, max_tokens, temperature):
    text = next(c["text"] for c in messages[0]["content"] if c["type"] == "text")
    words = text.split("Here are the candidate words:")[1].split("###")[0]
    ok = any(ch.isalpha() for ch in words)
    reply = json.dumps({"reasoning": "fake", "interpretable": ok,
                        "concrete_words": ["w"] if ok else [], "abstract_words": [],
                        "global_words": []})
    msg = types.SimpleNamespace(content="Thinking... " + reply)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def make_images(d, n):
  os.makedirs(os.path.join(d, "images"))
  rng = np.random.RandomState(0)
  meta = []
  for i in range(n):
    w, h = (400, 300) if i % 2 else (300, 420)
    Image.fromarray(rng.randint(0, 255, (h, w, 3), dtype=np.uint8)).save(
        os.path.join(d, "images", f"{i:05d}.jpg"))
    meta.append({"image_idx": i, "image_url": f"http://x/{i}", "caption": f"a cat number {i}"})
  with open(os.path.join(d, "images", "images.json"), "w") as f:
    json.dump(meta, f)


def make_bank(d, dim):
  rng = np.random.RandomState(1)
  for layer in (1, 2):
    os.makedirs(os.path.join(d, "bank", "lm", f"layer_{layer}"))
    meta = [{"token_str": "ing", "token_id": 292, "caption": "a rendering of a cat",
             "position": 2}] + [{"token_str": f"w{j}", "token_id": j, "caption": f"w{j} here",
                                 "position": 0} for j in range(1, 40)]
    torch.save({"embeddings": torch.tensor(rng.randn(40, dim), dtype=torch.float32),
                "metadata": meta, "token_to_indices": {}},
               os.path.join(d, "bank", "lm", f"layer_{layer}", "embeddings_cache.pt"))


def make_oracle(d, mt):
  ro = InjectionReadout(mt, ["object"], {"object": ["object"]}, "image", "cpu")
  add_lora(mt, 2, 4)
  for p in lora_parameters(mt).values():
    torch.nn.init.normal_(p, std=0.02)
  cfg = {"rank": 2, "lora_alpha": 4, "target_layer": 1, "no_norm_match": False, "inject": "add"}
  os.makedirs(d)
  torch.save({"lora": {n: p.detach().clone() for n, p in lora_parameters(mt).items()},
              "ref_norm": ro.ref_norm, "config": cfg}, os.path.join(d, "lora.pt"))
  set_lora(mt, False)


def main():
  stub_torchvision()
  import openai
  openai.OpenAI = FakeOpenAI
  R.LAYERS = (0, 1, 2, 4)  # the tiny model has 4 blocks
  mt, _ = build_tiny_vlm()
  n_img, n_patch = 6, 5
  with tempfile.TemporaryDirectory() as tmp:
    make_images(tmp, n_img)
    make_bank(tmp, mt.model.config.text_config.hidden_size)
    make_oracle(os.path.join(tmp, "oracle"), mt)
    base = ["--out_dir", tmp, "--ll_repo", LL_REPO, "--device", "cpu", "--n_images", str(n_img),
            "--num_patches", str(n_patch)]

    R.main(["--stage", "extract"] + base, mt=mt)
    lm = R.main(["--stage", "export_lm"] + base, mt=mt)
    ids = mt.tokenizer("a photo of a cat", return_tensors="pt")["input_ids"]
    with torch.no_grad():
      ref = mt.model(input_ids=ids).logits
      got = lm(input_ids=ids).logits
    assert torch.allclose(ref, got, atol=1e-4), "exported LM differs from LLaVA's"

    R.main(["--stage", "latentlens"] + base)
    R.main(["--stage", "plan"] + base)
    R.main(["--stage", "ours", "--oracle_dir", os.path.join(tmp, "oracle"), "--ps_target", "1",
            "--selfie_target", "1", "--slot_target", "1", "--rows_per_batch", "4"] + base, mt=mt)

    ev = R.ll_eval_module(R.parse_args(["--stage", "judge"] + base), "evaluate_interpretability")
    plan = json.load(open(os.path.join(tmp, "plan.json")))
    assert len(plan) == n_patch * len(R.LAYERS)
    for name, method in [("logitlens", "logitlens"), ("embeddinglens", "embeddinglens"),
                         ("latentlens", "latentlens"), ("patchscopes", "logitlens"),
                         ("selfie", "logitlens"), ("image_slot", "logitlens"),
                         ("oracle", "logitlens")]:
      data = ev.load_analysis_results(R.readout_dir(R.parse_args(base + ["--stage", "plan"]),
                                                    name), 2)
      assert ev.detect_method(data) == method, (name, ev.detect_method(data))
      images = ev.get_images_from_data(data, method)
      assert len(images) == n_img
      for p in plan:
        if p["layer"] != 2:
          continue
        img = next(x for x in images if x["image_idx"] == p["image_idx"])
        patch = next(q for q in img["chunks"][0]["patches"]
                     if (q["patch_row"], q["patch_col"]) == (p["row"], p["col"]))
        words = ev.extract_words_for_patch(patch, method)
        assert len(words) == 5, (name, words)
        if name == "latentlens" and "ing" in [n["token_str"] for n in
                                              patch["nearest_contextual_neighbors"]]:
          assert "rendering" in words and "ing" not in words

    R.main(["--stage", "judge"] + base)
    df = R.main(["--stage", "summary"] + base)
    assert set(df["readout"]) == set(R.THEIRS) | set(R.OURS)
    assert (df["n"] == n_patch).all() and (df["off_plan"] == 0).all()
    assert (df["unparsed"] == 0).all()
  print("\nLL PROTOCOL TEST PASSED")


if __name__ == "__main__":
  main()
