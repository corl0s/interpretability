"""CPU tests for the rewind maps (vtr/rewind.py, fit_rewind_maps.py, rewind variants).

  * ridge fit on streamed statistics recovers a known affine map (R^2 ~ 1)
  * fit_rewind_maps end to end on a tiny random LLaVA, never using benchmark images
  * run_injection_proto rewind variants: identity when source == target, different otherwise

  python test_rewind.py
"""

import json
import os
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fit_rewind_maps  # noqa: E402
import run_injection_proto  # noqa: E402
from test_visual_object_identification import build_tiny_vlm, make_fake_coco  # noqa: E402
from vtr.cache import cache_states  # noqa: E402
from vtr.data import select_object_samples  # noqa: E402
from vtr.geometry import Llava15Geometry  # noqa: E402
from vtr.rewind import PairStats, RewindMaps, fit_maps, r2  # noqa: E402


def test_recovers_affine_map():
  torch.manual_seed(0)
  d = 16
  W = torch.randn(d, d) / d ** 0.5
  b = torch.randn(d)
  stats = PairStats([(1, 0)], d, "cpu")
  def layer_pair(n):
    x = torch.randn(n, d) * 2 + 1
    return x, x @ W + b + 0.01 * torch.randn(n, d)
  for _ in range(20):
    x, y = layer_pair(50)
    stats.update(torch.stack([y, x]))  # layer 0 = y, layer 1 = x
  assert stats.n == 1000
  vx, vy = layer_pair(200)
  tx, ty = layer_pair(200)
  maps, df = fit_maps(stats, {0: vy, 1: vx}, {0: ty, 1: tx})
  Wf, bf = maps[(1, 0)]
  assert df.r2_test.iloc[0] > 0.99, df
  assert torch.allclose(Wf.float(), W, atol=0.02) and torch.allclose(bf, b, atol=0.05)
  assert r2(ty, ty) == 1.0
  print("ridge fit OK (recovers a known affine map)")


def add_unlabeled_images(coco_dir, n):
  """Extra val2017 images with no annotations (what the maps are fitted on)."""
  ann_path = os.path.join(coco_dir, "annotations", "instances_val2017.json")
  with open(ann_path, encoding="utf-8") as f:
    coco = json.load(f)
  rng = np.random.RandomState(1)
  for i in range(n):
    file_name = f"{1000 + i:012d}.jpg"
    Image.fromarray(rng.randint(0, 255, (300, 400, 3), dtype=np.uint8)).save(
        os.path.join(coco_dir, "val2017", file_name))
    coco["images"].append({"id": 1000 + i, "file_name": file_name, "height": 300, "width": 400})
  with open(ann_path, "w", encoding="utf-8") as f:
    json.dump(coco, f)


def main():
  test_recovers_affine_map()
  with tempfile.TemporaryDirectory() as tmp:
    coco_dir = os.path.join(tmp, "coco")
    ann_path = make_fake_coco(coco_dir)
    geometry = Llava15Geometry()
    samples, class_names = select_object_samples(ann_path, geometry, n_images=4, per_class=3,
                                                 min_area=20000, max_area=None, min_tokens=4,
                                                 seed=0)
    add_unlabeled_images(coco_dir, 6)
    mt, processor = build_tiny_vlm()
    states, meta = cache_states(mt, processor, samples, coco_dir, per_bin=2, n_outside=2,
                                seed=0)
    vtr_dir = os.path.join(tmp, "vtr")
    os.makedirs(vtr_dir)
    with open(os.path.join(vtr_dir, "samples.json"), "w", encoding="utf-8") as f:
      json.dump({"class_names": class_names, "samples": samples}, f)
    torch.save({"states": states, "meta": meta}, os.path.join(vtr_dir, "states.pt"))
    with open(os.path.join(vtr_dir, "fair_classes.json"), "w", encoding="utf-8") as f:
      json.dump({"kept": class_names, "dropped": {}}, f)

    fit_rewind_maps.main(["--coco_dir", coco_dir, "--vtr_dir", vtr_dir, "--device", "cpu",
                          "--sources", "0,2", "--targets", "0,1", "--n_images", "4",
                          "--n_heldout", "2", "--heldout_tokens", "8"], mt=mt)
    rdir = os.path.join(vtr_dir, "rewind")
    with open(os.path.join(rdir, "fit_images.json"), encoding="utf-8") as f:
      used = json.load(f)
    bench = {s["image_id"] for s in samples}
    assert not bench & set(used["fit"] + used["heldout"]), "maps saw a benchmark image"
    maps = RewindMaps(os.path.join(rdir, "maps.pt"))
    assert set(maps.maps) == {(0, 1), (2, 0), (2, 1)}, set(maps.maps)
    h = states[:3, 2].float()
    assert torch.equal(maps.apply(h, 0, 0), h)
    assert maps.apply(h, 2, 0).shape == h.shape
    print("fit_rewind_maps OK (benchmark images excluded, maps for every s != t)")

    out_dir = os.path.join(tmp, "proto")
    variants = ["img_all", "img_all_rewind", "selfie_text", "selfie_text_rewind"]
    run_injection_proto.main(["--vtr_dir", vtr_dir, "--out_dir", out_dir, "--device", "cpu",
                              "--sources", "0,2", "--targets", "0,1", "--n_images", "4",
                              "--n_boot", "20", "--dev_fraction", "0.5",
                              "--variants", *variants], mt=mt)
    cell = lambda v, s, t: np.load(os.path.join(out_dir, "cells", v, f"s{s}_t{t}.npz"))["pmi"]
    for base in ("img_all", "selfie_text"):
      assert np.array_equal(cell(base, 0, 0), cell(base + "_rewind", 0, 0))  # identity
      assert not np.allclose(cell(base, 2, 0), cell(base + "_rewind", 2, 0))
    print("rewind variants OK (identity at s == t, mapped otherwise)")
  print("\nALL REWIND TESTS PASSED")


if __name__ == "__main__":
  main()
