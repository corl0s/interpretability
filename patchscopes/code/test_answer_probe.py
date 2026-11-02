"""CPU test for answer_probe.py on synthetic data (no model):

  * the probe trains on the oracle's pool and labels answers by the fair class they name
  * held-out classes are never a probe output, so the probe's held-out top-1 is 0
  * on separable synthetic states the probe reads seen classes well
  * the oracle / untrained cells and the COCO-label probe are read and compared on the same rows
"""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import torch

import answer_probe as A
import run_vtr

CLASSES = ["dog", "cat", "car", "bus", "cup"]
HELD = ["bus"]
LAYERS = [0, 4]


def main():
  rng = np.random.RandomState(0)
  d, n_vtr_layers = 16, 6
  proto = rng.randn(len(CLASSES) + 1, d) * 3
  with tempfile.TemporaryDirectory() as tmp:
    vtr = os.path.join(tmp, "vtr")
    orc = os.path.join(vtr, "oracle")
    os.makedirs(os.path.join(orc, "data"))
    os.makedirs(os.path.join(vtr, "scores"))
    json.dump({"class_names": CLASSES, "samples": []}, open(os.path.join(vtr, "samples.json"), "w"))
    json.dump({"kept": CLASSES}, open(os.path.join(vtr, "fair_classes.json"), "w"))
    json.dump(HELD, open(os.path.join(orc, "holdout.json"), "w"))
    torch.save({"config": {"seed": 0, "target_layer": 4, "train_examples": 300,
                           "eval_images": 40, "data_dir": os.path.join(orc, "data")}},
               os.path.join(orc, "lora.pt"))

    # oracle training shard: answers name a class (some plural / held-out), are junk, or are other
    n_tok = 400
    which = rng.randint(0, len(CLASSES) + 1, n_tok)  # last index = background ("grass")
    answers = [("grass" if k == len(CLASSES) else CLASSES[k] + ("s" if i % 3 == 0 else ""))
               for i, k in enumerate(which)]
    answers[:5] = ["box"] * 5
    st = torch.tensor(proto[which][:, None, :] + rng.randn(n_tok, len(LAYERS), d),
                      dtype=torch.float16)
    torch.save({"states": st, "answers": answers, "image_id": list(range(n_tok)),
                "token": [0] * n_tok, "layers": LAYERS, "label_view": "crop7"},
               os.path.join(orc, "data", "shard_0000.pt"))

    # benchmark states: 40 images x 3 object rows
    rows = []
    for img in range(40):
      for j in range(3):
        rows.append({"image_id": img, "kind": "object", "coverage": 0.9,
                     "class_name": CLASSES[(img + j) % len(CLASSES)]})
    meta = pd.DataFrame(rows)
    lab = meta["class_name"].map({c: i for i, c in enumerate(CLASSES)}).to_numpy()
    states = torch.tensor(proto[lab][:, None, :] + rng.randn(len(meta), n_vtr_layers, d),
                          dtype=torch.float16)
    torch.save({"states": states, "meta": meta}, os.path.join(vtr, "states.pt"))

    # oracle / untrained cells and the COCO-label probe, on the select_rows order
    sel = A.select_rows(meta, 40, 0)
    for name, good in (("oracle", True), ("untrained", False)):
      cdir = os.path.join(orc, "eval", "cells", name)
      os.makedirs(cdir)
      for s in LAYERS:
        cell = rng.randn(len(sel), len(CLASSES))
        if good:
          cell[np.arange(len(sel)), lab[sel]] += 5
        np.savez(os.path.join(cdir, f"s{s}_t4.npz"), pmi=cell.astype(np.float16),
                 raw=cell.astype(np.float16))
    probe = rng.randn(n_vtr_layers, len(meta), len(CLASSES))
    probe[:, np.arange(len(meta)), lab] += 5
    run_vtr.save_scores(os.path.join(vtr, "scores", "probe_linear.npz"),
                        list(range(n_vtr_layers)), probe)

    A.main(["--vtr_dir", vtr, "--oracle_dir", orc])
    out = os.path.join(orc, "answer_probe")
    groups = pd.read_csv(os.path.join(out, "groups.csv"), index_col=0)
    table = pd.read_csv(os.path.join(out, "table.csv"), index_col=0)
    for r in ("answer_probe", "answer_probe_pooled"):
      assert groups.loc[r, "held-out"] == 0, groups
      assert groups.loc[r, "seen"] > 0.8, groups
    assert groups.loc["oracle", "held-out"] > 0.8
    assert {"oracle", "image_slot (untrained)", "probe_linear (COCO labels)"} <= set(table.index)
    # labels: plural forms map to the class, junk is dropped, held-out answers are dropped
    y = A.answer_labels(["dogs", "grass", "cup"], [0, 1, 2], CLASSES, HELD, CLASSES)
    assert list(y) == ["dog", A.OTHER, "cup"], y
    # second run reuses the saved scores
    A.main(["--vtr_dir", vtr, "--oracle_dir", orc])
  print("\nANSWER PROBE TEST PASSED")


if __name__ == "__main__":
  main()
