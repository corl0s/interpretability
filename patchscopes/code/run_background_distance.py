r"""Task 6 follow-up: is the background-token signal spillover or scene context? CPU only.

Background ("outside") tokens do not touch the target object, yet readouts decode the object from
them above chance. For each background token we compute its distance to the object on the token
grid (Chebyshev distance to the nearest token the object covers at all) and report every
readout's top-1 accuracy (against the object's class) by distance bin and layer:
  adjacent  distance 1  (touches an object token: vision-encoder / attention spillover)
  near      distance 2-3
  far       distance >= 4 (mostly scene context)

Outputs in <vtr_dir>/background_distance/: background_by_distance.csv, background_distance.png

  python run_background_distance.py
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_object_identification import bootstrap_by_image  # noqa: E402
from run_failure_map import SHORT, load_meta, load_scores  # noqa: E402
from vtr import evaluate as E  # noqa: E402

TOOLS = ["logit_lens", "latentlens", "patchscopes_pmi", "selfie_pmi", "tuned_lens",
         "embedding_lens", "probe_linear"]
BINS = (("adjacent", 1, 1), ("near", 2, 3), ("far", 4, 99))


def grid_distance(coverage, token, grid=24):
  """Chebyshev distance from `token` to the nearest token with coverage > 0."""
  cov = np.asarray(coverage).reshape(grid, grid)
  rows, cols = np.nonzero(cov > 0)
  if not len(rows):
    return np.nan
  r, c = divmod(int(token), grid)
  return int(np.max(np.abs(np.stack([rows - r, cols - c])), axis=0).min())


def distance_bin(d):
  for name, lo, hi in BINS:
    if lo <= d <= hi:
      return name
  return "other"


def parse_args(argv=None):
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--vtr_dir", default="./results/vtr_llava15")
  p.add_argument("--out_dir", default=None)
  p.add_argument("--all_classes", action="store_true")
  p.add_argument("--n_boot", type=int, default=1000)
  p.add_argument("--seed", type=int, default=0)
  return p.parse_args(argv)


def main(argv=None):
  args = parse_args(argv)
  out_dir = args.out_dir or os.path.join(args.vtr_dir, "background_distance")
  os.makedirs(out_dir, exist_ok=True)
  with open(os.path.join(args.vtr_dir, "samples.json"), encoding="utf-8") as f:
    data = json.load(f)
  class_names = data["class_names"]
  coverage = {s["image_id"]: s["coverage"] for s in data["samples"]}
  meta = load_meta(args.vtr_dir)
  scores = load_scores(args.vtr_dir, TOOLS)
  if not args.all_classes:
    with open(os.path.join(args.vtr_dir, "fair_classes.json"), encoding="utf-8") as f:
      kept = json.load(f)["kept"]
    scores, meta, class_names = E.subset(scores, meta, class_names,
                                         [class_names.index(c) for c in kept])

  outside = (meta["kind"] == "outside").to_numpy()
  dist = np.full(len(meta), np.nan)
  for i in np.flatnonzero(outside):
    dist[i] = grid_distance(coverage[meta["image_id"].iat[i]], meta["token"].iat[i])
  bins = np.array([distance_bin(d) if np.isfinite(d) else "" for d in dist])
  counts = pd.Series(bins[outside]).value_counts().to_dict()
  print(f"background tokens by distance: {counts}")

  index = {c: i for i, c in enumerate(class_names)}
  labels = meta["class_name"].map(index).to_numpy()
  frames = []
  for t, (layers, s) in scores.items():
    for name, _, _ in BINS:
      mask = outside & (bins == name) & ~np.isnan(s[0]).all(axis=1)
      if not mask.any():
        continue
      long = pd.DataFrame({
          "image_id": np.tile(meta["image_id"].to_numpy()[mask], len(layers)),
          "layer": np.repeat(layers, mask.sum()),
          "score": np.concatenate([(E.ranks(s[i], labels)[mask] == 1) for i in
                                   range(len(layers))]).astype(float)})
      b = bootstrap_by_image(long, "score", args.n_boot, args.seed)
      b["tool"], b["distance"], b["n_tokens"] = t, name, int(mask.sum())
      frames.append(b)
  out = pd.concat(frames, ignore_index=True)
  out.to_csv(os.path.join(out_dir, "background_by_distance.csv"), index=False)

  show = out[out.layer.isin([0, 2, 8, 16, 24, 30])].pivot_table(
      index=["tool", "distance"], columns="layer", values="mean").round(3)
  pd.set_option("display.width", 200)
  print(f"\nBackground-token top-1 accuracy (chance {1 / len(class_names):.3f}):")
  print(show.to_string())
  plot(out, os.path.join(out_dir, "background_distance.png"), len(class_names))
  print(f"\nAll outputs in {out_dir}")


def plot(out, path, n_classes):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  tools = [t for t in TOOLS if t in set(out.tool)]
  ncols = min(4, len(tools))
  nrows = int(np.ceil(len(tools) / ncols))
  fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.4 * nrows), sharey=True,
                           squeeze=False)
  styles = {"adjacent": ("#2a78d6", "-"), "near": ("#eb6834", "--"), "far": ("#1baf7a", ":")}
  for ax, t in zip(axes.flat, tools):
    for name, (colour, ls) in styles.items():
      g = out[(out.tool == t) & (out.distance == name)]
      if len(g):
        ax.plot(g.layer, g["mean"], ls, color=colour, lw=2, label=name)
    ax.axhline(1 / n_classes, color="#9a9994", lw=1, ls=":")
    ax.set_title(SHORT[t], loc="left", fontsize=10)
    ax.set_xlabel("layer")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", alpha=0.3)
    for side in ("top", "right"):
      ax.spines[side].set_visible(False)
  for ax in list(axes.flat)[len(tools):]:
    ax.set_visible(False)
  axes[0][0].set_ylabel("background tokens decoded as the object")
  axes[0][0].legend(frameon=False, fontsize=8, title="distance to object")
  fig.tight_layout()
  fig.savefig(path, dpi=150)
  plt.close(fig)


if __name__ == "__main__":
  main()
