"""CPU test for measure_teacher.py on a tiny random LLaVA: views, scoring, tables.

  python test_teacher.py
"""

import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import measure_teacher  # noqa: E402
from test_visual_object_identification import build_tiny_vlm, make_fake_coco  # noqa: E402
from vtr.cache import cache_states  # noqa: E402
from vtr.data import select_object_samples  # noqa: E402
from vtr.geometry import Llava15Geometry  # noqa: E402


def test_views():
  image = Image.new("RGB", (336, 336), (0, 0, 255))
  token = 24 * 5 + 7  # row 5, col 7
  crop, q = measure_teacher.make_view(image, token, "crop3")
  assert crop.size == (336, 336) and "main object" in q
  mask = np.array(measure_teacher.make_view(image, token, "mask5")[0])
  blue = np.argwhere(mask[..., 2] == 255)
  assert blue[:, 0].min() == 3 * 14 and blue[:, 0].max() == 8 * 14 - 1
  assert blue[:, 1].min() == 5 * 14 and blue[:, 1].max() == 10 * 14 - 1
  assert tuple(mask[0, 0]) == measure_teacher.GREY
  box, q = measure_teacher.make_view(image, 0, "box3")  # corner token: block clipped
  assert "red box" in q and tuple(np.array(box)[0, 0]) == (255, 0, 0)
  assert measure_teacher.block(0, 5) == (0, 3, 0, 3)
  print("views OK (crop, mask, box, clipping)")


def main():
  test_views()
  with tempfile.TemporaryDirectory() as tmp:
    coco_dir = os.path.join(tmp, "coco")
    ann_path = make_fake_coco(coco_dir)
    samples, class_names = select_object_samples(ann_path, Llava15Geometry(), n_images=4,
                                                 per_class=3, min_area=20000, max_area=None,
                                                 min_tokens=4, seed=0)
    mt, processor = build_tiny_vlm()
    states, meta = cache_states(mt, processor, samples, coco_dir, per_bin=2, n_outside=2, seed=0)
    vtr_dir = os.path.join(tmp, "vtr")
    os.makedirs(vtr_dir)
    with open(os.path.join(vtr_dir, "samples.json"), "w", encoding="utf-8") as f:
      json.dump({"class_names": class_names, "samples": samples}, f)
    torch.save({"states": states, "meta": meta}, os.path.join(vtr_dir, "states.pt"))
    with open(os.path.join(vtr_dir, "fair_classes.json"), "w", encoding="utf-8") as f:
      json.dump({"kept": class_names, "dropped": {}}, f)
    out_dir = os.path.join(tmp, "teacher")
    argv = ["--coco_dir", coco_dir, "--vtr_dir", vtr_dir, "--out_dir", out_dir, "--device", "cpu",
            "--views", "full,box3,crop5,mask5", "--batch", "3", "--n_boot", "20"]
    measure_teacher.main(argv, mt=mt)
    table = pd.read_csv(os.path.join(out_dir, "teacher_summary.csv"), index_col="view")
    assert list(table.index) == ["full", "box3", "crop5", "mask5"]
    assert table["closed"].between(0, 1).all() and table["free"].between(0, 1).all()
    assert table["same_as_full"].iloc[1:].notna().all()
    df = pd.read_csv(os.path.join(out_dir, "crop5.csv"), keep_default_na=False)
    assert (df.closed == (df.closed_pred == df.class_name)).all()
    measure_teacher.main(argv, mt=mt)  # second run reuses the saved views
  print("\nALL TEACHER TESTS PASSED")


if __name__ == "__main__":
  main()
