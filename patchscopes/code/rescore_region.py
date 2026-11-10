r"""Re-score the saved benchmark scores two more ways (CPU, no model).

  region   Takishita et al. (2506.05439) score the logit lens per object region: the label's rank
           after max-pooling the scores over the region's patches. Here, per image and layer, the
           class scores are max-pooled over the object's cached tokens (coverage >= 0.5; the
           benchmark caches up to 4 per image, so this is a sample of the region, not all of it;
           run_token_maps.py does the full region). Reported: top-1, top-5 and mean rank among
           the fair classes, next to the per-token top-1 of the benchmark.
  context  tokens the object does not touch (kind "outside"): the share whose top-1 is the
           image's object class. How far object information has spread into background tokens,
           per layer (chance = 1 / number of fair classes).

  python rescore_region.py --vtr_dir ./results/vtr_llava15
    -> <vtr_dir>/rescore/{region.csv, context.csv}
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_vtr import load_scores  # noqa: E402
from vtr import evaluate as E  # noqa: E402
from vtr.geometry import HIGH_COVERAGE  # noqa: E402

READOUTS = ["logit_lens", "tuned_lens", "embedding_lens", "latentlens", "patchscopes_pmi",
            "selfie_pmi", "image_slot_pmi", "probe_linear"]


def parse_args(argv=None):
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--vtr_dir", default="./results/vtr_llava15")
  p.add_argument("--readouts", default=",".join(READOUTS))
  return p.parse_args(argv)


def region_pool(scores, image_ids):
  """scores [n, C] for one image's rows each -> ([n_images, C] max over each image's rows,
  image ids in order)."""
  df = pd.DataFrame(scores)
  pooled = df.groupby(np.asarray(image_ids), sort=False).max()
  return pooled.to_numpy(), pooled.index.to_numpy()


def region_rows(name, layers, scores, rows, labels, image_ids, keep):
  out = []
  for li, l in enumerate(layers):
    s = scores[li][rows][:, keep]
    token_top1 = float((E.ranks(s, labels) == 1).mean())
    pooled, images = region_pool(s, image_ids)
    lab = pd.Series(labels).groupby(np.asarray(image_ids), sort=False).first().loc[images]
    r = E.ranks(pooled, lab.to_numpy())
    out.append({"readout": name, "layer": int(l), "token_top1": token_top1,
                "region_top1": float((r == 1).mean()), "region_top5": float((r <= 5).mean()),
                "region_mean_rank": float(r.mean()), "n_images": len(images)})
  return out


def context_rows(name, layers, scores, rows, labels, keep):
  out = []
  for li, l in enumerate(layers):
    hit = E.ranks(scores[li][rows][:, keep], labels) == 1
    out.append({"readout": name, "layer": int(l), "outside_names_object": float(hit.mean()),
                "n_tokens": int(len(rows))})
  return out


def main(argv=None):
  args = parse_args(argv)
  out_dir = os.path.join(args.vtr_dir, "rescore")
  os.makedirs(out_dir, exist_ok=True)
  with open(os.path.join(args.vtr_dir, "samples.json"), encoding="utf-8") as f:
    class_names = json.load(f)["class_names"]
  with open(os.path.join(args.vtr_dir, "fair_classes.json"), encoding="utf-8") as f:
    kept = json.load(f)["kept"]
  keep = [class_names.index(c) for c in kept]
  meta = torch.load(os.path.join(args.vtr_dir, "states.pt"), weights_only=False)["meta"]
  remap = {c: i for i, c in enumerate(kept)}
  fair = meta["class_name"].isin(kept)
  obj = np.flatnonzero((fair & (meta["kind"] == "object")
                        & (meta["coverage"] >= HIGH_COVERAGE)).to_numpy())
  out = np.flatnonzero((fair & (meta["kind"] == "outside")).to_numpy())

  region, context = [], []
  for name in args.readouts.split(","):
    path = os.path.join(args.vtr_dir, "scores", f"{name}.npz")
    if not os.path.exists(path):
      print(f"  (no {path}: {name} skipped)")
      continue
    layers, scores = load_scores(path)
    region += region_rows(name, layers, scores, obj,
                          meta["class_name"].iloc[obj].map(remap).to_numpy(),
                          meta["image_id"].iloc[obj].to_numpy(), keep)
    context += context_rows(name, layers, scores, out,
                            meta["class_name"].iloc[out].map(remap).to_numpy(), keep)
  region, context = pd.DataFrame(region), pd.DataFrame(context)
  region.to_csv(os.path.join(out_dir, "region.csv"), index=False)
  context.to_csv(os.path.join(out_dir, "context.csv"), index=False)

  pd.set_option("display.width", 200)
  for col in ("token_top1", "region_top1", "region_top5"):
    print(f"\n{col} by layer ({len(kept)} fair classes):")
    print(region.pivot(index="readout", columns="layer", values=col).round(2).to_string())
  print(f"\nOutside tokens whose top-1 is the image's object (chance {1 / len(kept):.3f}):")
  print(context.pivot(index="readout", columns="layer", values="outside_names_object")
        .round(2).to_string())
  print(f"\nAll outputs in {out_dir}")


if __name__ == "__main__":
  main()
