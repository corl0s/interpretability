r"""Task 2: per-tool failure map (complementarity) on the COCO-VTR scores. CPU only.

For every state (object token x layer) we record which tools rank the true class first (or in
the top 5). Per layer we then report, with image-clustered 95% CIs:
  acc_<tool>        accuracy of each tool
  only_<tool>       decoded by that tool and by no other tool (its unique contribution)
  union             decoded by at least one tool ("union coverage")
  best_single       accuracy of the best single tool at that layer
  gain              union - best_single (what combining tools could add at most)
  none              decoded by no tool ("decoding gap")
plus the share of every combination of tools (patterns.csv) and "A but not B" for every pair.

By default the analysis uses the fair class subset (fair_classes.json written by run_vtr.py
--stage eval), so every tool can score every candidate, and only the layers every tool was run
on. The supervised probe is reported as a ceiling, not as one of the tools.

Outputs in <vtr_dir>/failure_map/:
  failure_summary.csv, patterns.csv, pairwise.csv, failure_map[_top5|_pooled].png

  python run_failure_map.py                      # main map: logit lens, LatentLens, Patchscopes, SelfIE
  python run_failure_map.py --extended           # + tuned lens and embedding lens
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_object_identification import bootstrap_by_image  # noqa: E402
from vtr import evaluate as E  # noqa: E402
from vtr.geometry import BINS, HIGH_COVERAGE  # noqa: E402

MAIN_TOOLS = ["logit_lens", "latentlens", "patchscopes_pmi", "selfie_pmi"]
EXTRA_TOOLS = ["tuned_lens", "embedding_lens"]
class _Short(dict):
  """Display names; unknown score names (e.g. patchscopes_t0_pmi) fall back to readable text."""

  def __missing__(self, key):
    base, _, rest = key.partition("_t")
    if rest and rest.split("_")[0].isdigit() and base in ("patchscopes", "selfie"):
      return f"{self[base + '_pmi']}@L{rest.split('_')[0]}"
    return key


SHORT = _Short({"logit_lens": "LogitLens", "latentlens": "LatentLens",
                "patchscopes_pmi": "Patchscopes", "selfie_pmi": "SelfIE",
                "tuned_lens": "TunedLens", "embedding_lens": "EmbeddingLens",
                "probe_linear": "Probe"})


def complementarity(correct, tools):
  """correct: bool [n_rows, n_tools]. Returns per-row indicator columns for the summary."""
  n_correct = correct.sum(1)
  cols = {f"acc_{t}": correct[:, i] for i, t in enumerate(tools)}
  cols.update({f"only_{t}": correct[:, i] & (n_correct == 1) for i, t in enumerate(tools)})
  cols["union"] = n_correct > 0
  cols["shared"] = n_correct >= 2
  cols["none"] = n_correct == 0
  return {k: v.astype(float) for k, v in cols.items()}


def pattern_shares(correct, tools):
  """Share of rows for every combination of tools that decode them (exactly that set)."""
  keys = ["+".join(SHORT[t] for t, c in zip(tools, row) if c) or "none" for row in correct]
  return pd.Series(keys).value_counts(normalize=True)


def load_meta(vtr_dir):
  path = os.path.join(vtr_dir, "states.pt")
  try:
    cache = torch.load(path, weights_only=False, mmap=True)
  except (TypeError, RuntimeError):
    cache = torch.load(path, weights_only=False)
  return cache["meta"]


def load_scores(vtr_dir, names):
  out = {}
  for n in names:
    path = os.path.join(vtr_dir, "scores", f"{n}.npz")
    if os.path.exists(path):
      d = np.load(path)
      out[n] = (list(d["layers"]), d["scores"].astype(np.float32))
  return out


def conditions(meta):
  obj = (meta["kind"] == "object").to_numpy()
  conds = {"object_high": obj & (meta["coverage"] >= HIGH_COVERAGE).to_numpy(),
           "pooled_mean": (meta["kind"] == "pooled_mean").to_numpy()}
  for name, _, _ in BINS:
    conds[f"bin_{name}"] = obj & (meta["bin"] == name).to_numpy()
  return conds


def analyse(scores, meta, class_names, tools, n_boot, seed):
  index = {c: i for i, c in enumerate(class_names)}
  labels = meta["class_name"].map(index).to_numpy()
  layer_sets = [set(scores[t][0]) for t in tools]
  layers = sorted(set.intersection(*layer_sets))
  ranks = {}
  for t in list(tools) + (["probe_linear"] if "probe_linear" in scores else []):
    ls, s = scores[t]
    ranks[t] = {l: E.ranks(s[ls.index(l)], labels) for l in layers}

  # Only rows every tool actually scored (e.g. Patchscopes may cover a subset of images).
  scored = np.ones(len(meta), dtype=bool)
  for t in tools:
    ls, s = scores[t]
    scored &= ~np.isnan(s[[ls.index(l) for l in layers]]).all(axis=2).any(axis=0)

  summary, patterns, pairwise = [], [], []
  for cond, mask in conditions(meta).items():
    mask = mask & scored
    if not mask.any():
      continue
    image_ids = meta["image_id"].to_numpy()[mask]
    for metric, k in (("top1", 1), ("top5", 5)):
      per_layer = []
      for l in layers:
        correct = np.stack([ranks[t][l][mask] <= k for t in tools], 1)
        cols = complementarity(correct, tools)
        if "probe_linear" in ranks:
          cols["acc_probe_linear"] = (ranks["probe_linear"][l][mask] <= k).astype(float)
        per_layer.append(pd.DataFrame({"image_id": image_ids, "layer": l, **cols}))
        for pat, share in pattern_shares(correct, tools).items():
          patterns.append({"condition": cond, "metric": metric, "layer": l, "pattern": pat,
                           "share": round(float(share), 4)})
        for (i, a), (j, b) in itertools.permutations(list(enumerate(tools)), 2):
          pairwise.append({"condition": cond, "metric": metric, "layer": l, "tool": a,
                           "other": b,
                           "share_tool_not_other": float((correct[:, i] & ~correct[:, j]).mean())})
      long = pd.concat(per_layer, ignore_index=True)
      acc_cols = [f"acc_{t}" for t in tools]
      for col in [c for c in long.columns if c not in ("image_id", "layer")]:
        b = bootstrap_by_image(long, col, n_boot, seed)
        b["quantity"], b["condition"], b["metric"] = col, cond, metric
        summary.append(b)
      # best single tool per layer (by point estimate) and gain = union - best
      means = long.groupby("layer")[acc_cols + ["union"]].mean()
      best_tool = means[acc_cols].idxmax(axis=1)
      pick = long["layer"].map(best_tool).map(acc_cols.index).to_numpy()
      long["best_single"] = long[acc_cols].to_numpy()[np.arange(len(long)), pick]
      long["gain"] = long["union"] - long["best_single"]
      for col in ("best_single", "gain"):
        b = bootstrap_by_image(long, col, n_boot, seed)
        b["quantity"], b["condition"], b["metric"] = col, cond, metric
        if col == "best_single":
          b["best_tool"] = [best_tool[l].replace("acc_", "") for l in b["layer"]]
        summary.append(b)
  return (pd.concat(summary, ignore_index=True), pd.DataFrame(patterns),
          pd.DataFrame(pairwise), layers)


PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def plot(summary, tools, path, condition, metric, title):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  s = summary[(summary.condition == condition) & (summary.metric == metric)]
  piv = s.pivot_table(index="layer", columns="quantity", values="mean")
  layers = piv.index.to_numpy()
  fig, ax = plt.subplots(figsize=(11, 5.5))
  bottom = np.zeros(len(layers))
  for i, t in enumerate(tools):
    v = piv[f"only_{t}"].to_numpy()
    ax.bar(layers, v, bottom=bottom, width=1.6, color=PALETTE[i % len(PALETTE)],
           label=f"only {SHORT[t]}", edgecolor="white", linewidth=0.8)
    bottom += v
  v = piv["shared"].to_numpy()
  ax.bar(layers, v, bottom=bottom, width=1.6, color="#b8bcc4", label="2 or more tools",
         edgecolor="white", linewidth=0.8)
  ax.plot(layers, piv["union"], "k-", lw=2, label="union (any tool)")
  ax.plot(layers, piv["best_single"], "k--", lw=1.5, label="best single tool")
  if "acc_probe_linear" in piv:
    ax.plot(layers, piv["acc_probe_linear"], ":", color="#4a3aa7", lw=1.5,
            label="probe (supervised ceiling)")
  ax.set_xlabel("LLM layer")
  ax.set_ylabel(f"share of object states decoded ({metric})")
  ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
  ax.set_title(title, loc="left", fontsize=11)
  for side in ("top", "right"):
    ax.spines[side].set_visible(False)
  ax.grid(axis="y", alpha=0.3)
  ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
  fig.tight_layout()
  fig.savefig(path, dpi=150)
  plt.close(fig)


def parse_args(argv=None):
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--vtr_dir", default="./results/vtr_llava15")
  p.add_argument("--out_dir", default=None, help="default: <vtr_dir>/failure_map")
  p.add_argument("--extended", action="store_true", help="add tuned lens and embedding lens")
  p.add_argument("--tools", nargs="+", default=None,
                 help="score names to use instead of the default set, e.g. logit_lens "
                      "latentlens patchscopes_t0_pmi selfie_t0_pmi")
  p.add_argument("--all_classes", action="store_true", help="use all classes, not the fair subset")
  p.add_argument("--n_boot", type=int, default=1000)
  p.add_argument("--seed", type=int, default=0)
  return p.parse_args(argv)


def main(argv=None):
  args = parse_args(argv)
  out_dir = args.out_dir or os.path.join(args.vtr_dir, "failure_map" +
                                         ("_extended" if args.extended else ""))
  os.makedirs(out_dir, exist_ok=True)
  with open(os.path.join(args.vtr_dir, "samples.json"), encoding="utf-8") as f:
    class_names = json.load(f)["class_names"]
  meta = load_meta(args.vtr_dir)
  wanted = args.tools or (MAIN_TOOLS + (EXTRA_TOOLS if args.extended else []))
  scores = load_scores(args.vtr_dir, wanted + ["probe_linear"])
  tools = [t for t in wanted if t in scores]
  missing = [t for t in wanted if t not in scores]
  if missing:
    print(f"missing scores (skipped): {missing}")

  if not args.all_classes:
    with open(os.path.join(args.vtr_dir, "fair_classes.json"), encoding="utf-8") as f:
      kept = json.load(f)["kept"]
    keep = [class_names.index(c) for c in kept]
    scores, meta, class_names = E.subset(scores, meta, class_names, keep)
  print(f"tools: {tools}; {len(class_names)} classes; {meta['image_id'].nunique()} images")

  summary, patterns, pairwise, layers = analyse(scores, meta, class_names, tools,
                                                args.n_boot, args.seed)
  summary.to_csv(os.path.join(out_dir, "failure_summary.csv"), index=False)
  patterns.to_csv(os.path.join(out_dir, "patterns.csv"), index=False)
  pairwise.to_csv(os.path.join(out_dir, "pairwise.csv"), index=False)

  for cond, metric, suffix, label in (("object_high", "top1", "", "top-1"),
                                      ("object_high", "top5", "_top5", "top-5"),
                                      ("pooled_mean", "top1", "_pooled", "top-1, mean-pooled")):
    plot(summary, tools, os.path.join(out_dir, f"failure_map{suffix}.png"), cond, metric,
         f"Which tools decode each object state ({label}, {len(class_names)} classes)")

  view = summary[(summary.condition == "object_high") & (summary.metric == "top1")]
  table = view.pivot_table(index="layer", columns="quantity", values="mean").round(3)
  cols = ([f"acc_{t}" for t in tools] + [f"only_{t}" for t in tools]
          + ["union", "best_single", "gain", "none"]
          + (["acc_probe_linear"] if "acc_probe_linear" in table else []))
  table = table[cols].rename(columns=lambda c: c.replace("_pmi", "").replace("acc_", "acc:")
                             .replace("only_", "only:"))
  best = view[view.quantity == "best_single"].set_index("layer")["best_tool"]
  table["best_tool"] = best.reindex(table.index).str.replace("_pmi", "")
  pd.set_option("display.width", 250)
  print("\nObject states (coverage >= 50%), top-1:")
  print(table.to_string())
  print(f"\nAll outputs in {out_dir}")


if __name__ == "__main__":
  main()
