"""CPU tests for the injection prototype (vtr/injection.py, run_injection_proto.py).

  * text target: first-token scores equal Patchscopes' full scores for single-token continuations
  * image target: 576 contiguous image slots; own / 3x3 / footprint / all map to the right prompt
    positions; norm matching writes states of the reference norm; injection changes the scores;
    re-injection beyond the last layer is skipped
  * run_injection_proto end to end on a tiny random LLaVA (cells, dev/test selection, outputs)

  python test_injection.py
"""

import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_injection_proto  # noqa: E402
from test_visual_object_identification import build_tiny_vlm, make_fake_coco  # noqa: E402
from vtr.cache import cache_states  # noqa: E402
from vtr.data import select_object_samples  # noqa: E402
from vtr.geometry import Llava15Geometry  # noqa: E402
from vtr.injection import InjectionReadout  # noqa: E402
from vtr.readouts import Patchscopes, class_forms  # noqa: E402


def test_text_matches_patchscopes(mt, class_names, h):
  forms = class_forms(class_names, synonyms=False)
  ro = InjectionReadout(mt, class_names, forms, "text", "cpu", rows_per_batch=4)
  ps = Patchscopes(mt, class_names, forms, "selfie", "cpu", rows_per_batch=64)
  assert ro.pos == ps.pos and len(ro.pos) == 5
  single = (ps.cont_mask.sum(1) == 1).numpy()
  assert single.any()
  fast = ro._logp(h, [ro.pos] * len(h), 2).numpy()
  full = ps._seq_logp(h, 2).numpy()
  assert np.allclose(fast[:, single], full[:, single], atol=1e-4), \
      np.abs(fast[:, single] - full[:, single]).max()
  print("text target OK (first-token == Patchscopes for single-token continuations)")


def test_image_slots(mt, class_names, h):
  forms = class_forms(class_names, synonyms=False)
  ro = InjectionReadout(mt, class_names, forms, "image", "cpu", rows_per_batch=4)
  assert len(ro.pos) == 576 and ro.side == 24 and ro.pos[-1] - ro.pos[0] == 575
  assert ro.slots("own", 25) == [ro.pos[25]]
  block = ro.slots("own3x3", 25)
  assert sorted(block) == sorted(ro.pos[i] for i in (0, 1, 2, 24, 25, 26, 48, 49, 50))
  assert len(ro.slots("own3x3", 0)) == 4  # corner, clipped
  cov = np.zeros(576)
  cov[[30, 31, 54]] = [0.6, 1.0, 0.2]
  assert ro.slots("footprint", -1, cov) == [ro.pos[30], ro.pos[31]]
  assert ro.slots("own", -1, cov) == [ro.pos[30], ro.pos[31]]  # pooled rows fall back
  assert len(ro.slots("all")) == 576

  # Norm matching: the states written at layer 1 (read at layer 2's input) have the reference
  # norm.
  seen = {}
  layer = 1
  positions = [ro.slots("own3x3", 100)] * len(h)

  def probe(module, args, kwargs):
    hs = args[0] if args else kwargs["hidden_states"]
    seen["norm"] = hs[:, positions[0]].float().norm(dim=-1)
  handle = ro.layers[layer + 1].register_forward_pre_hook(probe, with_kwargs=True)
  ro._logp(h, positions, layer, norm_match=True)
  handle.remove()
  assert torch.allclose(seen["norm"], torch.full_like(seen["norm"], float(ro.ref_norm[layer])),
                        rtol=1e-3)

  raw, pmi = ro.scores(h, layer, [ro.slots("all")] * len(h))
  assert raw.shape == (len(h), len(class_names)) and np.isfinite(pmi).all()
  assert not np.allclose(pmi, 0, atol=1e-5)  # injection changes the scores
  a, _ = ro.scores(h, 3, [ro.slots("all")] * len(h), extra_layers=(4, 8))  # 3+4, 3+8 > 3 layers
  b, _ = ro.scores(h, 3, [ro.slots("all")] * len(h))
  assert np.allclose(a, b, atol=1e-5)
  print("image target OK (slots, norm matching, injection, multi-layer bounds)")


def main():
  with tempfile.TemporaryDirectory() as tmp:
    coco_dir = os.path.join(tmp, "coco")
    ann_path = make_fake_coco(coco_dir)
    geometry = Llava15Geometry()
    samples, class_names = select_object_samples(ann_path, geometry, n_images=4, per_class=3,
                                                 min_area=20000, max_area=None, min_tokens=4,
                                                 seed=0)
    mt, processor = build_tiny_vlm()
    states, meta = cache_states(mt, processor, samples, coco_dir, per_bin=2, n_outside=2,
                                seed=0)
    h = states[:3, 2].float()
    test_text_matches_patchscopes(mt, class_names, h)
    test_image_slots(mt, class_names, h)

    vtr_dir = os.path.join(tmp, "vtr")
    os.makedirs(vtr_dir)
    with open(os.path.join(vtr_dir, "samples.json"), "w", encoding="utf-8") as f:
      json.dump({"class_names": class_names, "samples": samples}, f)
    torch.save({"states": states, "meta": meta}, os.path.join(vtr_dir, "states.pt"))
    with open(os.path.join(vtr_dir, "fair_classes.json"), "w", encoding="utf-8") as f:
      json.dump({"kept": class_names, "dropped": {}}, f)
    out_dir = os.path.join(tmp, "proto")
    run_injection_proto.main(["--vtr_dir", vtr_dir, "--out_dir", out_dir, "--device", "cpu",
                              "--sources", "0,2", "--targets", "0,1", "--n_images", "4",
                              "--n_boot", "20", "--dev_fraction", "0.5"], mt=mt)
    for f in ("proto_summary.csv", "proto_best.csv", "proto_by_source.png"):
      assert os.path.exists(os.path.join(out_dir, f)), f
    n_cells = sum(len(fs) for _, _, fs in os.walk(os.path.join(out_dir, "cells")))
    assert n_cells == len(run_injection_proto.VARIANTS) * 4, n_cells
    print("run_injection_proto OK")
  print("\nALL INJECTION TESTS PASSED")


if __name__ == "__main__":
  main()
