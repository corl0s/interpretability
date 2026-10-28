r"""Compare image-slot + LatentLens (slot_nn) with its parts on the same rows.

slot_nn runs on the first --n_images benchmark images (run_vtr.py --ps_max_images), so every
readout here is restricted to those images and to the fair classes (fair_classes.json), then
evaluated exactly like run_vtr's eval stage. Readouts missing from <out_dir>/scores are skipped.

  python compare_slot_nn.py --out_dir ./results/vtr_llava15 --bank_name latentlens_vg
    -> <out_dir>/slot_nn/compare_<bank_name>.csv and a top-1 table (object tokens) per layer
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_vtr import load_scores  # noqa: E402
from vtr import evaluate as E  # noqa: E402


def main(argv=None):
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--out_dir", default="./results/vtr_llava15")
  p.add_argument("--bank_name", default="latentlens_vg")
  p.add_argument("--n_images", type=int, default=200)
  p.add_argument("--n_boot", type=int, default=500)
  p.add_argument("--seed", type=int, default=0)
  args = p.parse_args(argv)

  with open(os.path.join(args.out_dir, "samples.json"), encoding="utf-8") as f:
    class_names = json.load(f)["class_names"]
  with open(os.path.join(args.out_dir, "fair_classes.json"), encoding="utf-8") as f:
    kept = json.load(f)["kept"]
  meta = torch.load(os.path.join(args.out_dir, "states.pt"), weights_only=False)["meta"]
  keep_imgs = meta["image_id"].drop_duplicates().iloc[:args.n_images]
  rows = meta["image_id"].isin(keep_imgs).to_numpy()

  score_dir = os.path.join(args.out_dir, "scores")
  prefix = f"slot_nn_{args.bank_name}_"
  names = ["latentlens", args.bank_name, "image_slot_pmi", "oracle_pmi", "logit_lens",
           "selfie_pmi", "probe_linear"]
  names += sorted(f[:-4] for f in os.listdir(score_dir)
                  if f.startswith(prefix) and f.endswith(".npz"))
  scores = {}
  for n in names:
    path = os.path.join(score_dir, f"{n}.npz")
    if os.path.exists(path):
      ls, s = load_scores(path)
      scores[n] = (ls, s[:, rows])
  sub_meta = meta[rows].reset_index(drop=True)
  keep = [class_names.index(c) for c in kept]
  sub, sub_meta, sub_names = E.subset(scores, sub_meta, class_names, keep)
  print(f"{sub_meta['image_id'].nunique()} images, {len(sub_meta)} rows, {len(sub_names)} "
        f"classes; readouts: {sorted(sub)}")
  summary = E.evaluate(sub, sub_meta, sub_names, args.n_boot, args.seed)
  out = os.path.join(args.out_dir, "slot_nn")
  os.makedirs(out, exist_ok=True)
  summary.to_csv(os.path.join(out, f"compare_{args.bank_name}.csv"), index=False)

  # Average over the layers slot_nn was run on, so every row's mean covers the same layers.
  nn_layers = next((ls for n, (ls, _) in scores.items() if n.startswith(prefix)), None)
  top1 = summary[(summary.metric == "top1")]
  if nn_layers is not None:
    top1 = top1[top1.layer.isin(nn_layers)]
  for cond in ("object_high", "outside"):
    view = top1[top1.condition == cond].pivot_table(index="readout", columns="layer",
                                                    values="mean")
    view.insert(0, "mean", view.mean(axis=1))
    print(f"\nTop-1 (%), {cond}:")
    print((view * 100).round(1).sort_values("mean", ascending=False).to_string())
  print(f"\nAll outputs in {out}")


if __name__ == "__main__":
  main()
