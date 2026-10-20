r"""Fit label-free rewind maps (vtr/rewind.py) from image-token states of unlabeled COCO images.

Images: COCO val2017 images NOT used by the benchmark (samples.json), so the maps never see an
evaluation image. Every image token of every fitting image is used (576 per LLaVA image). No
annotation or class label is read except to list the image files.

For every (source, target) pair with source != target the script fits h_t ~ h_s @ W + b. The
ridge strength is chosen on half of the held-out images and R^2 is reported on the other half,
next to the identity baseline (inject the state unchanged).

Outputs in --out_dir:
  maps.pt          {"maps": {(s, t): (W fp16 [d, d], b fp32 [d])}, "sources", "targets", ...}
  rewind_fit.csv   per pair: alpha, R^2 (val, test), identity R^2, norm ratio
  fit_images.json  the image ids used for fitting and held-out evaluation

  python fit_rewind_maps.py --coco_dir /projectnb/mlresearch/vishnuav/coco \
      --vtr_dir ./results/vtr_llava15                                    # ~30-60 min on a GPU
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_vtr  # noqa: E402
from visual_object_identification import ensure_annotations, ensure_image  # noqa: E402
from vtr.models import capture_states  # noqa: E402
from vtr.rewind import PairStats, fit_maps, map_pairs  # noqa: E402


def parse_args(argv=None):
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--coco_dir", required=True)
  p.add_argument("--vtr_dir", default="./results/vtr_llava15",
                 help="benchmark dir; its samples.json images are excluded")
  p.add_argument("--out_dir", default=None, help="default: <vtr_dir>/rewind")
  p.add_argument("--model_name", default="llava-hf/llava-1.5-7b-hf")
  p.add_argument("--sources", default="0,4,8,12,16,20,24,28")
  p.add_argument("--targets", default="0,4,8,12,16")
  p.add_argument("--n_images", type=int, default=2000, help="images used to fit the maps")
  p.add_argument("--n_heldout", type=int, default=200, help="images for alpha choice and R^2")
  p.add_argument("--heldout_tokens", type=int, default=64, help="tokens kept per held-out image")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  return p.parse_args(argv)


def pick_images(coco_dir, vtr_dir, n, seed):
  with open(ensure_annotations(coco_dir), encoding="utf-8") as f:
    images = json.load(f)["images"]
  with open(os.path.join(vtr_dir, "samples.json"), encoding="utf-8") as f:
    used = {s["image_id"] for s in json.load(f)["samples"]}
  pool = sorted((im for im in images if im["id"] not in used), key=lambda im: im["id"])
  if len(pool) < n:
    raise ValueError(f"only {len(pool)} unused images, asked for {n}")
  idx = np.random.RandomState(seed).permutation(len(pool))[:n]
  return [pool[i] for i in idx]


def main(argv=None, mt=None):
  args = parse_args(argv)
  out_dir = args.out_dir or os.path.join(args.vtr_dir, "rewind")
  os.makedirs(out_dir, exist_ok=True)
  sources = [int(x) for x in args.sources.split(",")]
  targets = [int(x) for x in args.targets.split(",")]
  pairs = map_pairs(sources, targets)

  vtr_args = run_vtr.parse_args(["--model_name", args.model_name, "--coco_dir", args.coco_dir,
                                 "--device", args.device])
  geometry = run_vtr.make_geometry(vtr_args)
  if mt is None:
    mt = run_vtr.load_model(vtr_args)

  images = pick_images(args.coco_dir, args.vtr_dir, args.n_images + args.n_heldout, args.seed)
  fit_imgs, held_imgs = images[:args.n_images], images[args.n_images:]
  with open(os.path.join(out_dir, "fit_images.json"), "w", encoding="utf-8") as f:
    json.dump({"fit": [im["id"] for im in fit_imgs], "heldout": [im["id"] for im in held_imgs]},
              f)
  print(f"{len(pairs)} maps; fitting on {len(fit_imgs)} images, held out {len(held_imgs)}")

  def states_of(im, verify=False):
    image = Image.open(ensure_image(args.coco_dir, im["file_name"])).convert("RGB")
    return capture_states(mt, mt.processor, image, geometry, verify=verify)  # [L, T, d]

  rng = np.random.RandomState(args.seed + 1)
  stats, t0 = None, time.time()
  for i, im in enumerate(fit_imgs):
    full = states_of(im, verify=(i == 0))
    if stats is None:
      stats = PairStats(pairs, full.shape[-1], args.device)
    stats.update(full)
    if (i + 1) % 100 == 0 or i == 0:
      print(f"  fit images {i + 1}/{len(fit_imgs)} ({stats.n} tokens, "
            f"{time.time() - t0:.0f}s)", flush=True)

  held = {l: [] for l in stats.layers}
  for im in held_imgs:
    full = states_of(im)
    keep = rng.choice(full.shape[1], min(args.heldout_tokens, full.shape[1]), replace=False)
    for l in stats.layers:
      held[l].append(full[l, keep])
  half = len(held_imgs) // 2
  val = {l: torch.cat(v[:half]) for l, v in held.items()}
  test = {l: torch.cat(v[half:]) for l, v in held.items()}

  maps, df = fit_maps(stats, val, test)
  torch.save({"maps": maps, "sources": sources, "targets": targets,
              "model_name": args.model_name, "n_train_tokens": stats.n},
             os.path.join(out_dir, "maps.pt"))
  df.to_csv(os.path.join(out_dir, "rewind_fit.csv"), index=False)
  print("\nTest R^2 of the rewind map (identity baseline in brackets):")
  for t in targets:
    line = [f"{r.r2_test:.2f} ({r.r2_test_identity:.2f})"
            for r in df[df.target == t].itertuples()]
    print(f"  to layer {t:2d}: " + "  ".join(line))
  print(f"\nAll outputs in {out_dir}")


if __name__ == "__main__":
  main()
