"""CPU test for diagnose_readouts.py on a fake COCO (two images, three classes):

  * image 1: target dog, a larger car next to it (dominant -> the dog is "hard"); image 2: a cat
  * a readout that names the dog on the car's outside token is counted as reading the image
  * a readout that answers "car" on the hard dog token is counted as naming the dominant object
  * the probe is right on every object token, so agreement and edge / size tables fill in
"""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import torch

import diagnose_readouts as D
import run_vtr
from vtr.geometry import GEOMETRIES

CLASSES = ["dog", "cat", "car"]
SUPER = {"dog": "animal", "cat": "animal", "car": "vehicle"}


def square(x0, y0, x1, y1):
  return [[x0, y0, x1, y0, x1, y1, x0, y1]]


def make_coco(coco_dir):
  os.makedirs(os.path.join(coco_dir, "annotations"))
  anns = [  # image 1: dog (left), car (right, larger); image 2: cat
      {"id": 10, "image_id": 1, "category_id": 1, "segmentation": square(0, 0, 40, 100),
       "area": 4000.0, "iscrowd": 0},
      {"id": 11, "image_id": 1, "category_id": 3, "segmentation": square(50, 0, 100, 100),
       "area": 5000.0, "iscrowd": 0},
      {"id": 20, "image_id": 2, "category_id": 2, "segmentation": square(0, 0, 50, 50),
       "area": 2500.0, "iscrowd": 0},
  ]
  coco = {"images": [{"id": i, "file_name": f"{i}.jpg", "height": 100, "width": 100}
                     for i in (1, 2)],
          "annotations": anns,
          "categories": [{"id": k + 1, "name": c, "supercategory": SUPER[c]}
                         for k, c in enumerate(CLASSES)]}
  json.dump(coco, open(os.path.join(coco_dir, "annotations", "instances_val2017.json"), "w"))
  return {a["id"]: a for a in anns}


def test_diagnose():
  from visual_object_identification import ann_to_mask
  geo = GEOMETRIES["llava-1.5"]()
  with tempfile.TemporaryDirectory() as tmp:
    coco_dir, vtr = os.path.join(tmp, "coco"), os.path.join(tmp, "vtr")
    os.makedirs(os.path.join(vtr, "scores"))
    anns = make_coco(coco_dir)
    samples, meta = [], []
    for image_id, ann_id, cls in ((1, 10, "dog"), (2, 20, "cat")):
      cov = geo.coverage(ann_to_mask(anns[ann_id], 100, 100))
      samples.append({"image_id": image_id, "file_name": f"{image_id}.jpg", "class_name": cls,
                      "supercategory": SUPER[cls], "ann_id": ann_id, "area": 1.0,
                      "size": "medium", "coverage": [float(c) for c in cov], "grid": [24, 24]})
      base = {"image_id": image_id, "class_name": cls, "supercategory": SUPER[cls],
              "size": "medium"}
      on = int(np.flatnonzero(cov >= 0.99)[0])
      off = 23 if image_id == 1 else 575  # on the car / background
      assert cov[off] == 0
      meta += [{**base, "kind": "object", "token": on, "coverage": float(cov[on]), "bin": "75-100"},
               {**base, "kind": "outside", "token": off, "coverage": 0.0, "bin": "outside"},
               {**base, "kind": "pooled_mean", "token": -1, "coverage": np.nan,
                "bin": "pooled_mean"}]
    json.dump({"class_names": CLASSES, "samples": samples}, open(os.path.join(vtr, "samples.json"), "w"))
    meta = pd.DataFrame(meta)
    torch.save({"states": torch.zeros(len(meta), 2, 4), "meta": meta},
               os.path.join(vtr, "states.pt"))

    def scores(pick):
      s = np.zeros((2, len(meta), len(CLASSES)))
      for i, m in meta.iterrows():
        s[:, i, CLASSES.index(pick(m))] = 1.0
      return s

    leak = scores(lambda m: m["class_name"])            # names the target everywhere
    dominant = scores(lambda m: "car" if m["image_id"] == 1 else m["class_name"])
    run_vtr.save_scores(os.path.join(vtr, "scores", "leak.npz"), [0, 1], leak)
    run_vtr.save_scores(os.path.join(vtr, "scores", "dominant.npz"), [0, 1], dominant)
    run_vtr.save_scores(os.path.join(vtr, "scores", "probe_linear.npz"), [0, 1], leak)

    D.main(["--vtr_dir", vtr, "--coco_dir", coco_dir,
            "--readouts", "leak,dominant,probe_linear,missing"])
    out = os.path.join(vtr, "diagnose")
    refs = pd.read_csv(os.path.join(out, "refs.csv"))
    r1 = refs[(refs.image_id == 1) & (refs.kind == "outside")].iloc[0]
    assert r1["dominant"] == "car" and r1["local"] == "car"
    cond = pd.read_csv(os.path.join(out, "conditions.csv"), header=[0, 1], index_col=0)
    assert cond.loc["leak", ("outside_on_object", "early")] == 1.0
    assert cond.loc["dominant", ("outside_local", "early")] == 1.0
    assert cond.loc["leak", ("outside_background", "early")] == 1.0
    hard = pd.read_csv(os.path.join(out, "hardness.csv"), header=[0, 1], index_col=0)
    assert hard.loc["dominant", ("hard", "early")] == 0.0
    assert hard.loc["dominant", ("easy", "early")] == 1.0
    err = pd.read_csv(os.path.join(out, "errors_hard.csv"))
    d = err[err.readout == "dominant"]
    assert (d["dominant"] == 1.0).all() and (d["chance_dominant"] == 0.5).all()
    agree = pd.read_csv(os.path.join(out, "agreement.csv"))
    assert (agree["probe_and_other"] + agree["all"] == 1.0).all()
    # second run reads refs.csv back
    D.main(["--vtr_dir", vtr, "--coco_dir", coco_dir, "--readouts", "leak,probe_linear"])


def main():
  test_diagnose()
  print("\nDIAGNOSE READOUTS TEST PASSED")


if __name__ == "__main__":
  main()
