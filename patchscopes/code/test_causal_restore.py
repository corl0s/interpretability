"""CPU test for causal_restore.py (no model):

  * masking paints the (dilated) object grey and finds the tokens it touches
  * restore_states overwrites one layer's output at the chosen positions, per batch row only
  * capture_image_states stores every layer's image-token states
  * readout_hits + case_tables sort (image, layer) into A/B/C/D, and summarize runs end to end
"""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

import causal_restore as C
import run_vtr
from vtr.oracle import GREY

KEPT = ["dog", "cat", "car"]


class AddOne(nn.Module):
  def forward(self, x):
    return (x + 1,)


def test_masking():
  image = Image.new("RGB", (40, 40), (255, 0, 0))
  mask = np.zeros((40, 40), bool)
  mask[18:22, 18:22] = True
  out, grown = C.mask_object(image, mask, dilate=2)
  a = np.array(out)
  assert (a[grown] == GREY).all() and (a[~grown] == (255, 0, 0)).all()
  assert grown.sum() == 8 * 8 and grown[mask].all()
  m = np.zeros((336, 336), bool)
  m[:28, :28] = True  # the top-left 2x2 patches of the 24x24 grid
  assert list(C.restore_positions(m)) == [0, 1, 24, 25]


def test_hooks():
  layers = nn.ModuleList([AddOne() for _ in range(3)])
  x = torch.zeros(3, 6, 2)
  start = 2
  plan = [(1, torch.tensor([0, 2]), torch.full((2, 2), 7.0)), None,
          (0, torch.tensor([1]), torch.full((1, 2), 5.0))]

  def run(x):
    for layer in layers:
      x = layer(x)[0]
    return x

  with C.restore_states(layers, start, plan):
    y = run(x)
  assert torch.all(y[1] == 3)                                   # untouched row
  assert torch.all(y[0, [2, 4]] == 8) and torch.all(y[0, [0, 1, 3, 5]] == 3)  # 7 after layer 1
  assert torch.all(y[2, 3] == 7) and torch.all(y[2, 2] == 3)    # 5 after layer 0
  assert torch.all(run(x) == 3)                                 # hooks removed
  store = {}
  with C.capture_image_states(layers, start, 3, store):
    run(x)
  assert sorted(store) == [0, 1, 2] and store[1].shape == (3, 2) and torch.all(store[1] == 2)


def test_summary():
  rng = np.random.RandomState(0)
  with tempfile.TemporaryDirectory() as tmp:
    vtr = os.path.join(tmp, "vtr")
    out = os.path.join(vtr, "causal_restore")
    os.makedirs(os.path.join(vtr, "scores"))
    os.makedirs(out)
    json.dump({"class_names": KEPT, "samples": []}, open(os.path.join(vtr, "samples.json"), "w"))
    # 3 images x 2 object tokens (+ one low-coverage token that must be ignored)
    meta = pd.DataFrame([{"image_id": i, "kind": "object", "coverage": cov, "class_name": "dog"}
                         for i in range(3) for cov in (0.9, 0.8, 0.1)])
    torch.save({"states": torch.zeros(len(meta), 2, 4), "meta": meta},
               os.path.join(vtr, "states.pt"))
    # readout "good" decodes every image at layer 0, none at layer 1; "bad" never decodes
    good = rng.randn(2, len(meta), 3)
    good[0, :, 0] += 10
    good[1, :, 0] -= 10
    bad = rng.randn(2, len(meta), 3)
    bad[:, :, 0] -= 10
    run_vtr.save_scores(os.path.join(vtr, "scores", "good.npz"), [0, 1], good)
    run_vtr.save_scores(os.path.join(vtr, "scores", "bad.npz"), [0, 1], bad)
    # image 2 is invalid (masking does not hurt); used at layer 0 only
    pd.DataFrame({"image_id": [0, 1, 2], "class_name": "dog", "n_restored": 5,
                  "p_clean": [0.9, 0.8, 0.9], "p_masked": [0.1, 0.1, 0.85],
                  "top1_clean": True, "top1_masked": [False, False, True],
                  "masked_answer": ["cat", "car", "dog"]}).to_csv(
        os.path.join(out, "images.csv"), index=False)
    pd.DataFrame([{"image_id": i, "mode": m, "layer": l, "p": 0.0, "top1": l == 0,
                   "recovery": 0.9 if l == 0 else 0.1}
                  for i in range(3) for m in ("object", "all_image") for l in (0, 1)]).to_csv(
        os.path.join(out, "restores.csv"), index=False)

    C.main(["--stage", "summarize", "--vtr_dir", vtr, "--readouts", "good,bad,missing"])
    cases = pd.read_csv(os.path.join(out, "cases.csv"), index_col=0)
    # good: layer 0 A, layer 1 D; bad: layer 0 C, layer 1 D
    assert cases.loc["good", "A"] == 0.5 and cases.loc["bad", "C"] == 0.5
    assert cases.loc["good", "hit_used"] == 1.0 and cases.loc["good", "hit_unused"] == 0.0
    assert cases.loc["bad", "hit_used"] == 0.0 and cases.loc["bad", "hit_unused"] == 0.0
    assert (cases["n"] == 4).all() and (cases["n_used"] == 2).all()  # 2 valid images x 2 layers
    curve = pd.read_csv(os.path.join(out, "curve.csv"))
    assert set(curve["mode"]) == {"object", "all_image"}


class Cycle:
  """Stands in for a fitted probe: predicts classes 0, 1, 2, 0, ... over the rows."""

  def predict(self, X):
    return np.arange(len(X)) % len(KEPT)


def test_square_mask():
  m = np.zeros((100, 200), bool)
  m[:, 100:] = True                       # right half; square crop keeps the middle 100 px
  sq = C.square_mask(m)
  assert sq.shape == (336, 336) and sq[:, 170:].all() and not sq[:, :166].any()
  g = C.grid_coverage(sq).reshape(24, 24)
  assert (g[:, 13:] == 1).all() and (g[:, :11] == 0).all()


def test_fit_probes():
  rng = np.random.RandomState(0)
  with tempfile.TemporaryDirectory() as tmp:
    meta = pd.DataFrame([{"image_id": i, "kind": "object", "coverage": 0.9,
                          "class_name": KEPT[i % 3]} for i in range(12)])
    states = torch.tensor(rng.randn(12, 2, 4), dtype=torch.float16)
    states[:, :, 0] += torch.tensor([3.0 * (i % 3) for i in range(12)])[:, None]
    torch.save({"states": states, "meta": meta}, os.path.join(tmp, "states.pt"))
    probes = C.fit_probes(tmp, tmp, KEPT, [0, 1], [0, 1])
    assert sorted(probes) == [0, 1]
    assert set(probes[0].classes_) == {0, 1, 2}
    again = C.fit_probes(tmp, tmp, KEPT, [0], [1])          # cached: [0] is a subset of [0, 1]
    assert again[1] is not None and sorted(again) == [0, 1]


def test_run_tiny():
  """stage_run end to end on a tiny random LLaVA: every restore mode and both experiments."""
  import train_oracle
  from test_visual_object_identification import build_tiny_vlm
  mt, _ = build_tiny_vlm()
  rng = np.random.RandomState(0)
  with tempfile.TemporaryDirectory() as tmp:
    coco, vtr = os.path.join(tmp, "coco"), os.path.join(tmp, "vtr")
    for d in ("annotations", "val2017"):
      os.makedirs(os.path.join(coco, d))
    os.makedirs(vtr)
    images, anns, samples = [], [], []
    for i in range(4):
      Image.fromarray(rng.randint(0, 255, (336, 336, 3), dtype=np.uint8)).save(
          os.path.join(coco, "val2017", f"{i}.jpg"))
      x0 = 40 * i
      images.append({"id": i, "file_name": f"{i}.jpg", "height": 336, "width": 336})
      anns.append({"id": 100 + i, "image_id": i, "category_id": i % 3 + 1, "iscrowd": 0,
                   "segmentation": [[x0, 40, x0 + 150, 40, x0 + 150, 200, x0, 200]],
                   "area": 150.0 * 160})
      samples.append({"image_id": i, "file_name": f"{i}.jpg", "class_name": KEPT[i % 3],
                      "ann_id": 100 + i})
    json.dump({"images": images, "annotations": anns,
               "categories": [{"id": k + 1, "name": c, "supercategory": "x"}
                              for k, c in enumerate(KEPT)]},
              open(os.path.join(coco, "annotations", "instances_val2017.json"), "w"))
    json.dump({"class_names": KEPT, "samples": samples}, open(os.path.join(vtr, "samples.json"), "w"))
    train_oracle.load_model = lambda args: mt
    C.fit_probes = lambda *a, **k: {l: Cycle() for l in range(4)}
    out = os.path.join(vtr, "causal_restore")
    C.main(["--stage", "run", "--coco_dir", coco, "--vtr_dir", vtr, "--n_images", "3",
            "--device", "cpu", "--batch", "4",
            "--modes", "object,all_image,context,decoded,undecoded,random_k",
            "--token_layers", "0,1", "--experiments", "masked_read,paste", "--read_layers", "0,3"])
    res = pd.read_csv(os.path.join(out, "restores.csv"))
    assert res["image_id"].nunique() == 3
    for m in ("object", "all_image", "context"):
      assert sorted(res[res["mode"] == m]["layer"].unique()) == [0, 1, 2, 3]
    per = res[res["layer"] == 0].pivot_table(index="image_id", columns="mode", values="n_tokens")
    assert (per["object"] + per["context"] == 576).all() and (per["all_image"] == 576).all()
    assert (per["decoded"] == per["undecoded"]).all() and (per["random_k"] == per["decoded"]).all()
    assert set(res[res["mode"].isin(C.TOKEN_MODES)]["layer"]) == {0, 1}
    reads = pd.read_csv(os.path.join(out, "reads.csv"))
    mr = reads[reads["experiment"] == "masked_read"]
    assert set(mr["image"]) == {"clean", "masked"} and set(mr["region"]) == {"object", "outside"}
    assert set(mr["layer"]) == {0, 3} and set(mr["reader"]) == {"probe", "logit_lens"}
    ps = reads[reads["experiment"] == "paste"]
    assert set(ps["region"]) == {"object", "scene"} and ps["scene"].notna().all()
    paste = pd.read_csv(os.path.join(out, "paste.csv"))
    assert (paste["donor_class"] != paste["class_name"]).all() and len(paste) == 3
    # rerun resumes (nothing left to do); the summaries read the new outputs
    C.main(["--stage", "run", "--coco_dir", coco, "--vtr_dir", vtr, "--n_images", "3",
            "--device", "cpu"])
    assert len(pd.read_csv(os.path.join(out, "restores.csv"))) == len(res)
    ids = set(res["image_id"])
    C.summarize_tokens(out, res, 0)
    C.summarize_reads(out, ids, ids)
    assert os.path.exists(os.path.join(out, "tokens.csv"))
    assert os.path.exists(os.path.join(out, "masked_read.csv"))
    assert os.path.exists(os.path.join(out, "paste_read.csv"))
    # an output directory from the older version is refused
    old = os.path.join(tmp, "old")
    os.makedirs(old)
    pd.DataFrame({"image_id": [0], "mode": ["object"], "layer": [0], "p": [0.1], "top1": [False],
                  "recovery": [0.0]}).to_csv(os.path.join(old, "restores.csv"), index=False)
    try:
      C.main(["--stage", "run", "--coco_dir", coco, "--vtr_dir", vtr, "--out_dir", old,
              "--device", "cpu"])
      raise AssertionError("old restores.csv must be refused")
    except SystemExit:
      pass


def main():
  test_masking()
  test_hooks()
  test_summary()
  test_square_mask()
  test_fit_probes()
  test_run_tiny()
  print("\nCAUSAL RESTORE TEST PASSED")


if __name__ == "__main__":
  main()
