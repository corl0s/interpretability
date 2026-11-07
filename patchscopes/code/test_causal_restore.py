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
    # good: layer 0 A, layer 1 D -> agreement 1; bad: layer 0 C, layer 1 D -> 0.5
    assert cases.loc["good", "agreement"] == 1.0 and cases.loc["good", "A"] == 0.5
    assert cases.loc["bad", "C"] == 0.5 and cases.loc["bad", "agreement"] == 0.5
    assert (cases["n"] == 4).all()  # 2 valid images x 2 layers
    curve = pd.read_csv(os.path.join(out, "curve.csv"))
    assert set(curve["mode"]) == {"object", "all_image"}


def main():
  test_masking()
  test_hooks()
  test_summary()
  print("\nCAUSAL RESTORE TEST PASSED")


if __name__ == "__main__":
  main()
