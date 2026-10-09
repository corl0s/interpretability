r"""Two checks on the unified decoder (Task 4). CPU only, existing scores.

1. Held-out classes ("why not just a probe?")
   The fusion has only per-tool temperatures and weights, nothing class-specific, so it should
   transfer to classes it never saw. For each of --repeats random splits of the classes into
   halves A and B:
     fit on dev images of classes A  ->  test on test images of classes B   (unseen classes)
     fit on dev images of classes B  ->  test on the same rows              (seen, reference)
   Candidates are always all classes of the (fair) set. A probe trained on A cannot output B at
   all, so on B it scores 0 by construction.

2. Label-free fusion ("it still uses labels")
   zscore_equal: each tool's scores are standardised per row across the candidates and the
   z-scores are averaged. No labels, no fitting. Compared on the same test images with the
   label-fitted fusion and the best single tool.

Outputs in <vtr_dir>/decoder_checks/: heldout_classes.csv, label_free.csv (per layer, plus a
"mean" row = mean over layers), printed summaries.

  python run_decoder_checks.py
  python run_decoder_checks.py --tools selfie_pmi latentlens logit_lens
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_unified_decoder as U  # noqa: E402
from run_failure_map import EXTRA_TOOLS, MAIN_TOOLS, load_meta, load_scores  # noqa: E402
from vtr import evaluate as E  # noqa: E402
from vtr.geometry import HIGH_COVERAGE  # noqa: E402


def prepare(scores, meta, class_names, tools, dev_fraction, seed):
  index = {c: i for i, c in enumerate(class_names)}
  labels = meta["class_name"].map(index).to_numpy()
  layers = sorted(set.intersection(*[set(scores[t][0]) for t in tools]))
  dev = U.split_images(meta, dev_fraction, seed)
  obj = ((meta["kind"] == "object") & (meta["coverage"] >= HIGH_COVERAGE)).to_numpy()
  scored = np.ones(len(meta), dtype=bool)
  for t in tools:
    ls, s = scores[t]
    scored &= ~np.isnan(s[[ls.index(l) for l in layers]]).all(axis=2).any(axis=0)
  raw = {l: {t: np.nan_to_num(scores[t][1][scores[t][0].index(l)], nan=0.0) for t in tools}
         for l in layers}
  return labels, layers, dev, obj & scored, raw


def fused_ranks(raw_l, labels, fit, tools):
  """Weighted and equal fusion fitted on rows `fit`; returns ranks for all rows."""
  temps = {t: U.fit_temperature(raw_l[t][fit], labels[fit]) for t in tools}
  logp = {t: U.log_softmax_t(raw_l[t], temps[t]) for t in tools}
  w = U.fit_weights([logp[t][fit] for t in tools], labels[fit])
  return (E.ranks(sum(wk * logp[t] for wk, t in zip(w, tools)), labels),
          E.ranks(np.mean([logp[t] for t in tools], 0), labels))


def heldout_classes(labels, layers, dev, usable, raw, tools, n_classes, repeats, seed):
  rows = []
  test = ~dev & usable
  for r in range(repeats):
    perm = np.random.RandomState(seed + 100 + r).permutation(n_classes)
    a, b = perm[: n_classes // 2], perm[n_classes // 2:]
    in_a, in_b = np.isin(labels, a), np.isin(labels, b)
    fit_a, fit_b, eval_b = dev & usable & in_a, dev & usable & in_b, test & in_b
    for l in layers:
      acc = {t: (E.ranks(raw[l][t], labels)[eval_b] == 1).mean() for t in tools}
      w_a, e_a = fused_ranks(raw[l], labels, fit_a, tools)
      w_b, _ = fused_ranks(raw[l], labels, fit_b, tools)
      rows.append({"repeat": r, "layer": l, "n_test_rows": int(eval_b.sum()),
                   "best_single_tool": max(acc.values()),
                   "best_tool": max(acc, key=acc.get),
                   "fusion_weighted_unseen_classes": (w_a[eval_b] == 1).mean(),
                   "fusion_equal_unseen_classes": (e_a[eval_b] == 1).mean(),
                   "fusion_weighted_seen_classes": (w_b[eval_b] == 1).mean(),
                   "probe_trained_on_other_classes": 0.0})
  df = pd.DataFrame(rows)
  num = [c for c in df.columns if c not in ("repeat", "layer", "best_tool", "n_test_rows")]
  per_layer = df.groupby("layer")[num].mean()
  per_layer_sd = df.groupby("layer")[num].std().add_suffix("_sd")
  out = pd.concat([per_layer, per_layer_sd], axis=1)
  out.loc["mean"] = out.mean()
  return out.round(4), df


def zscore(s):
  return (s - s.mean(axis=1, keepdims=True)) / (s.std(axis=1, keepdims=True) + 1e-8)


def label_free(labels, layers, dev, usable, raw, tools):
  test = ~dev & usable
  fit = dev & usable
  rows = []
  for l in layers:
    acc = {t: (E.ranks(raw[l][t], labels)[test] == 1).mean() for t in tools}
    zr = E.ranks(np.mean([zscore(raw[l][t]) for t in tools], 0), labels)
    wr, er = fused_ranks(raw[l], labels, fit, tools)
    rows.append({"layer": l, "best_single_tool": max(acc.values()),
                 "best_tool": max(acc, key=acc.get),
                 "zscore_equal_label_free": (zr[test] == 1).mean(),
                 "fusion_equal_fitted_T": (er[test] == 1).mean(),
                 "fusion_weighted_fitted": (wr[test] == 1).mean()})
  df = pd.DataFrame(rows).set_index("layer")
  df.loc["mean"] = df.drop(columns="best_tool").mean()
  df.loc["mean", "best_tool"] = ""
  return df


def parse_args(argv=None):
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--vtr_dir", default="./results/vtr_llava15")
  p.add_argument("--out_dir", default=None, help="default: <vtr_dir>/decoder_checks")
  p.add_argument("--tools", nargs="+", default=None)
  p.add_argument("--extended", action="store_true")
  p.add_argument("--all_classes", action="store_true")
  p.add_argument("--dev_fraction", type=float, default=0.3)
  p.add_argument("--repeats", type=int, default=5)
  p.add_argument("--seed", type=int, default=0)
  return p.parse_args(argv)


def main(argv=None):
  args = parse_args(argv)
  out_dir = args.out_dir or os.path.join(args.vtr_dir, "decoder_checks")
  os.makedirs(out_dir, exist_ok=True)
  with open(os.path.join(args.vtr_dir, "samples.json"), encoding="utf-8") as f:
    class_names = json.load(f)["class_names"]
  meta = load_meta(args.vtr_dir)
  wanted = args.tools or (MAIN_TOOLS + (EXTRA_TOOLS if args.extended else []))
  scores = load_scores(args.vtr_dir, wanted)
  tools = [t for t in wanted if t in scores]
  if not args.all_classes:
    with open(os.path.join(args.vtr_dir, "fair_classes.json"), encoding="utf-8") as f:
      kept = json.load(f)["kept"]
    scores, meta, class_names = E.subset(scores, meta, class_names,
                                         [class_names.index(c) for c in kept])
  labels, layers, dev, usable, raw = prepare(scores, meta, class_names, tools,
                                             args.dev_fraction, args.seed)
  print(f"tools: {tools}; {len(class_names)} classes; layers {layers}")
  pd.set_option("display.width", 220)

  held, held_all = heldout_classes(labels, layers, dev, usable, raw, tools, len(class_names),
                                   args.repeats, args.seed)
  held.to_csv(os.path.join(out_dir, "heldout_classes.csv"))
  held_all.to_csv(os.path.join(out_dir, "heldout_classes_all_repeats.csv"), index=False)
  cols = ["best_single_tool", "fusion_weighted_unseen_classes", "fusion_equal_unseen_classes",
          "fusion_weighted_seen_classes", "probe_trained_on_other_classes"]
  print(f"\n1. Held-out classes (fit on half the classes, test on the other half; "
        f"mean of {args.repeats} splits; test images, object tokens, top-1):")
  print(held[cols].to_string())

  lf = label_free(labels, layers, dev, usable, raw, tools)
  lf.to_csv(os.path.join(out_dir, "label_free.csv"))
  print("\n2. Label-free fusion (test images, object tokens, top-1):")
  print(lf.round(4).to_string())
  print(f"\nAll outputs in {out_dir}")


if __name__ == "__main__":
  main()
