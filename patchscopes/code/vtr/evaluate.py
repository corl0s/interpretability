"""Turn readout scores into accuracies per layer and condition, with image-clustered CIs.

Ranks are pessimistic: a class tied with the true class counts as ranked above it, so a
readout cannot score by giving every class the same value. A true class the readout cannot
score at all (-inf) is ranked last.

Conditions (rows of the cache):
  object_high      object tokens with coverage >= 0.5 (headline)
  bin_<a-b>        object tokens per overlap bin
  pooled_mean/max  pooled object states
  outside, random  controls
  shuffled         object_high rows scored against another image's label (prior-only chance)
"""

import numpy as np
import pandas as pd

from analyze_object_identification import bootstrap_by_image
from vtr.geometry import BINS, HIGH_COVERAGE


def ranks(scores, labels):
  """scores [n, C], labels [n] -> 1-based pessimistic rank of the true class."""
  true = scores[np.arange(len(labels)), labels][:, None]
  higher = (scores > true).sum(1)
  ties = (scores == true).sum(1) - 1
  r = 1 + higher + ties
  r[~np.isfinite(true[:, 0])] = scores.shape[1]
  return r


def condition_masks(meta):
  obj = (meta["kind"] == "object").to_numpy()
  masks = {"object_high": obj & (meta["coverage"] >= HIGH_COVERAGE).to_numpy()}
  for name, _, _ in BINS:
    masks[f"bin_{name}"] = obj & (meta["bin"] == name).to_numpy()
  for kind in ("pooled_mean", "pooled_max", "outside", "random"):
    masks[kind] = (meta["kind"] == kind).to_numpy()
  return masks


def shuffled_labels(meta, labels, seed):
  """Each image's label replaced by the label of a different image (a derangement)."""
  images = meta["image_id"].unique()
  rng = np.random.RandomState(seed)
  perm = rng.permutation(len(images))
  donor = dict(zip(images[perm], images[np.roll(perm, 1)]))
  label_of = dict(zip(meta["image_id"], labels))
  return meta["image_id"].map(lambda i: label_of[donor[i]]).to_numpy()


def evaluate(scores_by_readout, meta, class_names, n_boot, seed):
  """scores_by_readout: {name: (layers, array [n_layers, n_rows, C])} -> long DataFrame.

  Rows a readout did not score (all NaN, e.g. Patchscopes on a subset of images) are dropped
  for that readout only.
  """
  index = {c: i for i, c in enumerate(class_names)}
  labels = meta["class_name"].map(index).to_numpy()
  wrong = shuffled_labels(meta, labels, seed)
  masks = condition_masks(meta)
  frames = []
  for name, (layers, scores) in scores_by_readout.items():
    layers = np.asarray(layers)
    scored = ~np.isnan(scores[0]).all(axis=1)
    rk = np.stack([ranks(s, labels) for s in scores])          # [L, n]
    rk_shuf = np.stack([ranks(s, wrong) for s in scores])
    conds = dict(masks)
    conds["shuffled"] = masks["object_high"]
    for cond, mask in conds.items():
      mask = mask & scored
      if not mask.any():
        continue
      r = (rk_shuf if cond == "shuffled" else rk)[:, mask]
      for metric, k in (("top1", 1), ("top5", 5)):
        long = pd.DataFrame({
            "image_id": np.tile(meta["image_id"].to_numpy()[mask], len(layers)),
            "layer": np.repeat(layers, mask.sum()),
            "score": (r <= k).reshape(-1).astype(float),
        })
        frame = bootstrap_by_image(long, "score", n_boot, seed)
        frame["readout"], frame["condition"], frame["metric"] = name, cond, metric
        frame["n_rows"] = int(mask.sum())
        frames.append(frame)
  return pd.concat(frames, ignore_index=True)


def headline(summary):
  """Per readout: best layer, accuracy there and at layer 2, and the onset layer.

  Onset = first layer whose CI lower bound exceeds the highest control mean
  (outside / random / shuffled) of that readout at that layer.
  """
  rows = []
  top1 = summary[summary.metric == "top1"]
  for name, g in top1.groupby("readout"):
    obj = g[g.condition == "object_high"].set_index("layer")
    ctrl = (g[g.condition.isin(["outside", "random", "shuffled"])]
            .groupby("layer")["mean"].max())
    above = obj.index[obj["ci_lo"] > ctrl.reindex(obj.index).fillna(0)]
    best = int(obj["mean"].idxmax())
    pooled = g[g.condition == "pooled_mean"].set_index("layer")["mean"]
    rows.append({
        "readout": name,
        "best_layer": best,
        "acc_best": round(float(obj.loc[best, "mean"]), 4),
        "acc_layer2": round(float(obj.loc[2, "mean"]), 4) if 2 in obj.index else None,
        "onset_layer": int(above.min()) if len(above) else None,
        "pooled_mean_best": round(float(pooled.max()), 4) if len(pooled) else None,
        "max_control": round(float(ctrl.max()), 4) if len(ctrl) else None,
    })
  return pd.DataFrame(rows)


# Validated categorical slots (CVD-safe in this order).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def plot(summary, out_dir, main_readouts):
  import os

  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  top1 = summary[summary.metric == "top1"]
  colours = {r: PALETTE[i % len(PALETTE)] for i, r in enumerate(sorted(top1.readout.unique()))}

  def style(ax):
    for side in ("top", "right"):
      ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))

  # 1. Accuracy by layer, object tokens (coverage >= 0.5), main readouts.
  fig, ax = plt.subplots(figsize=(10, 5.5))
  for r in main_readouts:
    g = top1[(top1.readout == r) & (top1.condition == "object_high")]
    if len(g):
      ax.plot(g.layer, g["mean"], color=colours[r], lw=2, label=r)
      ax.fill_between(g.layer, g.ci_lo, g.ci_hi, color=colours[r], alpha=0.14, lw=0)
  ctrl = (top1[top1.readout.isin(main_readouts)
               & top1.condition.isin(["outside", "random", "shuffled"])]
          .groupby("layer")["mean"].max())
  ax.plot(ctrl.index, ctrl.values, ":", color="#9a9994", lw=1.5, label="max control")
  ax.set_xlabel("LLM layer (output of block l)")
  ax.set_ylabel("top-1 accuracy (80 classes)")
  ax.set_title("COCO-VTR objects, LLaVA-1.5-7B: readout accuracy by layer", loc="left")
  ax.legend(frameon=False, fontsize=9)
  style(ax)
  fig.tight_layout()
  fig.savefig(os.path.join(out_dir, "accuracy_by_layer.png"), dpi=150)
  plt.close(fig)

  # 2. Accuracy by overlap bin at each readout's best layer.
  fig, ax = plt.subplots(figsize=(7, 4.5))
  bins = [f"bin_{b[0]}" for b in BINS]
  for r in main_readouts:
    g = top1[top1.readout == r]
    obj = g[g.condition == "object_high"]
    if not len(obj):
      continue
    best = int(obj.loc[obj["mean"].idxmax(), "layer"])
    at = g[g.layer == best].set_index("condition")
    vals = [at.loc[b, "mean"] if b in at.index else np.nan for b in bins]
    ax.plot(range(len(bins)), vals, marker="o", color=colours[r], lw=2,
            label=f"{r} (layer {best})")
  ax.set_xticks(range(len(bins)))
  ax.set_xticklabels([b[0] + "%" for b in BINS])
  ax.set_xlabel("share of the token's region covered by the object")
  ax.set_ylabel("top-1 accuracy")
  ax.set_title("Decodability vs overlap ratio", loc="left")
  ax.legend(frameon=False, fontsize=8)
  style(ax)
  fig.tight_layout()
  fig.savefig(os.path.join(out_dir, "accuracy_by_overlap.png"), dpi=150)
  plt.close(fig)

  # 3. Single token vs pooled, per readout (small multiples).
  n = len(main_readouts)
  fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.6), sharey=True)
  axes = np.atleast_1d(axes)
  for ax, r in zip(axes, main_readouts):
    g = top1[top1.readout == r]
    for cond, ls, label in (("object_high", "-", "single token"),
                            ("pooled_mean", "--", "mean-pooled"),
                            ("pooled_max", ":", "max-pooled")):
      c = g[g.condition == cond]
      if len(c):
        ax.plot(c.layer, c["mean"], ls, color=colours[r], lw=2, label=label)
    ax.set_title(r, loc="left", fontsize=10)
    ax.set_xlabel("layer")
    style(ax)
  axes[0].set_ylabel("top-1 accuracy")
  axes[0].legend(frameon=False, fontsize=8)
  fig.tight_layout()
  fig.savefig(os.path.join(out_dir, "pooling.png"), dpi=150)
  plt.close(fig)
