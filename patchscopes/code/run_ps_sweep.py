r"""Task 5: Patchscopes source-layer x target-layer sweep on the COCO-VTR cache.

The state of a visual token at source layer s is injected into the Patchscopes prompt at target
layer t (same-layer patching is the diagonal s = t) and scored closed-set exactly as in
run_vtr.py (PMI-corrected log-likelihood of " -> <class>" over the 80 COCO classes).

Question: is Patchscopes weak at late layers because the information is missing, or because
same-layer injection leaves only a few layers to use it? If late source states score higher
when injected at an earlier target layer, the late-layer drop is a setup artefact.

Rows: for a subset of images, object tokens with coverage >= 0.5, the mean-pooled object state
and the norm-matched random control. Every cell is checkpointed, so a stopped job resumes.

Outputs in --out_dir:
  cells/s{S}_t{T}.npz      raw and PMI scores per cell
  sweep_summary.csv        accuracy per (source, target, condition, metric) with image CIs
  sweep_best_target.csv    best target layer per source layer vs the diagonal
  sweep_heatmap.png        top-1 accuracy, object tokens (and fair-subset version)

  python run_ps_sweep.py                                    # 200 images, every 4th layer + 31
  python run_ps_sweep.py --layers 0,2,4,...,30 --n_images 300
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_object_identification import bootstrap_by_image  # noqa: E402
from vtr import evaluate as E  # noqa: E402
from vtr.geometry import HIGH_COVERAGE  # noqa: E402
from vtr.readouts import Patchscopes, class_forms  # noqa: E402


def parse_args(argv=None):
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--vtr_dir", default="./results/vtr_llava15",
                 help="run_vtr.py output with states.pt and samples.json")
  p.add_argument("--out_dir", default="./results/vtr_llava15/ps_sweep")
  p.add_argument("--model_name", default="llava-hf/llava-1.5-7b-hf")
  p.add_argument("--layers", default="0,4,8,12,16,20,24,28,31",
                 help="used for both source and target layers")
  p.add_argument("--n_images", type=int, default=200)
  p.add_argument("--prompt", default="identity", choices=["identity", "entity", "selfie"])
  p.add_argument("--rows_per_batch", type=int, default=512)
  p.add_argument("--n_boot", type=int, default=1000)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  return p.parse_args(argv)


def select_rows(meta, n_images, seed):
  """Object tokens (coverage >= 0.5), pooled_mean and random rows of a random image subset."""
  images = meta["image_id"].drop_duplicates()
  chosen = images.sample(n=min(n_images, len(images)), random_state=seed)
  in_subset = meta["image_id"].isin(chosen).to_numpy()
  wanted = (((meta["kind"] == "object") & (meta["coverage"] >= HIGH_COVERAGE))
            | meta["kind"].isin(["pooled_mean", "random"])).to_numpy()
  return np.flatnonzero(in_subset & wanted)


def run_cells(args, mt, states, rows, class_names, layers):
  cell_dir = os.path.join(args.out_dir, "cells")
  os.makedirs(cell_dir, exist_ok=True)
  ro = Patchscopes(mt, class_names, class_forms(class_names, synonyms=False), args.prompt,
                   args.device, rows_per_batch=args.rows_per_batch)
  for s in layers:
    h = states[rows, s].float()
    for t in layers:
      path = os.path.join(cell_dir, f"s{s}_t{t}.npz")
      if os.path.exists(path):
        continue
      t0 = time.time()
      raw, pmi = ro.scores(h, t)
      np.savez(path, raw=raw.astype(np.float16), pmi=pmi.astype(np.float16))
      print(f"  source {s:2d} -> target {t:2d}: {len(rows)} rows ({time.time() - t0:.1f}s)",
            flush=True)


def summarize(args, meta_rows, class_names, layers, keep=None, tag=""):
  """Accuracy per cell and condition with image-clustered CIs."""
  index = {c: i for i, c in enumerate(class_names)}
  labels = meta_rows["class_name"].map(index).to_numpy()
  if keep is not None:
    ok = np.isin(labels, keep)
    remap = {c: i for i, c in enumerate(keep)}
  frames = []
  for s in layers:
    for t in layers:
      cell = np.load(os.path.join(args.out_dir, "cells", f"s{s}_t{t}.npz"))
      for variant in ("pmi", "raw"):
        scores = cell[variant].astype(np.float32)
        lab, m = labels, meta_rows
        if keep is not None:
          scores = scores[ok][:, keep]
          lab = np.array([remap[x] for x in labels[ok]])
          m = meta_rows[ok]
        rank = E.ranks(scores, lab)
        for cond in ("object", "pooled_mean", "random"):
          mask = (m["kind"] == cond).to_numpy()
          for metric, k in (("top1", 1), ("top5", 5)):
            df = pd.DataFrame({"image_id": m["image_id"].to_numpy()[mask], "layer": 0,
                               "score": (rank[mask] <= k).astype(float)})
            if not len(df):
              continue
            b = bootstrap_by_image(df, "score", args.n_boot, args.seed).iloc[0]
            frames.append({"source": s, "target": t, "variant": variant, "condition": cond,
                           "metric": metric, "mean": b["mean"], "ci_lo": b["ci_lo"],
                           "ci_hi": b["ci_hi"], "n_rows": int(mask.sum())})
  out = pd.DataFrame(frames)
  out.to_csv(os.path.join(args.out_dir, f"sweep_summary{tag}.csv"), index=False)

  obj = out[(out.variant == "pmi") & (out.condition == "object") & (out.metric == "top1")]
  best = []
  for s, g in obj.groupby("source"):
    top = g.loc[g["mean"].idxmax()]
    diag = g[g.target == s].iloc[0]
    best.append({"source": s, "best_target": int(top["target"]),
                 "acc_best_target": round(float(top["mean"]), 4),
                 "acc_same_layer": round(float(diag["mean"]), 4),
                 "gain": round(float(top["mean"] - diag["mean"]), 4)})
  best = pd.DataFrame(best)
  best.to_csv(os.path.join(args.out_dir, f"sweep_best_target{tag}.csv"), index=False)
  plot_heatmap(obj, layers, os.path.join(args.out_dir, f"sweep_heatmap{tag}.png"),
               len(keep) if keep is not None else len(class_names))
  return out, best


def plot_heatmap(obj, layers, path, n_classes):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  grid = obj.pivot(index="source", columns="target", values="mean").reindex(
      index=layers, columns=layers)
  fig, ax = plt.subplots(figsize=(7.5, 6.2))
  im = ax.imshow(grid.values, cmap="Blues", origin="lower", vmin=0)
  for i in range(len(layers)):
    for j in range(len(layers)):
      v = grid.values[i, j]
      if np.isfinite(v):
        ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=8,
                color="white" if v > 0.6 * np.nanmax(grid.values) else "#17191c")
  ax.set_xticks(range(len(layers)))
  ax.set_xticklabels(layers)
  ax.set_yticks(range(len(layers)))
  ax.set_yticklabels(layers)
  ax.set_xlabel("target layer (where the state is injected)")
  ax.set_ylabel("source layer (where the state is taken from)")
  ax.set_title(f"Patchscopes top-1 accuracy (%), object tokens, {n_classes} classes\n"
               "diagonal = same-layer patching", loc="left", fontsize=10)
  fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
               format=matplotlib.ticker.PercentFormatter(1.0, decimals=0))
  fig.tight_layout()
  fig.savefig(path, dpi=150)
  plt.close(fig)


def main(argv=None, mt=None):
  args = parse_args(argv)
  os.makedirs(args.out_dir, exist_ok=True)
  layers = [int(x) for x in args.layers.split(",")]
  with open(os.path.join(args.vtr_dir, "samples.json"), encoding="utf-8") as f:
    class_names = json.load(f)["class_names"]
  cache = torch.load(os.path.join(args.vtr_dir, "states.pt"), weights_only=False)
  states, meta = cache["states"], cache["meta"]
  rows = select_rows(meta, args.n_images, args.seed)
  meta_rows = meta.iloc[rows].reset_index(drop=True)
  print(f"{meta_rows['image_id'].nunique()} images, {len(rows)} rows "
        f"({meta_rows['kind'].value_counts().to_dict()}); "
        f"{len(layers)} x {len(layers)} cells, prompt={args.prompt}")
  with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
    json.dump({**vars(args), "rows": rows.tolist()}, f)

  if mt is None:
    from general_utils import ModelAndTokenizer
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    mt = ModelAndTokenizer(args.model_name, torch_dtype=dtype, device=args.device)
  run_cells(args, mt, states, rows, class_names, layers)

  _, best = summarize(args, meta_rows, class_names, layers)
  print("\nBest target layer per source layer (PMI, top-1, object tokens, all classes):")
  print(best.to_string(index=False))
  fair_path = os.path.join(args.vtr_dir, "fair_classes.json")
  if os.path.exists(fair_path):
    with open(fair_path, encoding="utf-8") as f:
      kept = json.load(f)["kept"]
    keep = [class_names.index(c) for c in kept]
    _, best_fair = summarize(args, meta_rows, class_names, layers, keep=keep, tag="_fair")
    print(f"\nSame, fair subset ({len(keep)} classes):")
    print(best_fair.to_string(index=False))
  print(f"\nAll outputs in {args.out_dir}")


if __name__ == "__main__":
  main()
