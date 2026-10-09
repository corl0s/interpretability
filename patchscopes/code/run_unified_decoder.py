r"""Task 4: a unified decoder that combines the readouts, evaluated on held-out images. CPU only.

Images are split once into a dev part (choices: calibration temperatures, fusion weights, routing)
and a test part (every reported number), so "beats every single tool" is not built in.

Per tool and layer, scores over the candidate classes are turned into calibrated
log-probabilities, log_softmax(score / T), with T fitted on dev (minimum NLL of the true class).

Decoders:
  fusion_equal     mean of the tools' calibrated log-probs (training-free apart from calibration)
  fusion_weighted  per-layer non-negative tool weights fitted on dev (K numbers per layer)
  rank_fusion      reciprocal-rank fusion, sum_k 1 / (60 + rank_k); no fitting (exact ties
                   broken by the mean calibrated log-prob)
  routing          per layer, the single tool with the best dev accuracy
References on test: every single tool, best_single_test (the best tool chosen on test: a strict,
slightly optimistic baseline), and union (some tool is right: the upper bound from Task 2).

Outputs in <vtr_dir>/unified/:
  unified_summary.csv     accuracy per method / layer / condition / metric with image CIs
  unified_headline.csv    per method: mean accuracy over layers (aggregate score), gain over
                          the best single tool, share of the possible gain (union) recovered
  unified_weights.csv     calibration temperatures, fusion weights and routing choice per layer
  unified_by_layer.png

  python run_unified_decoder.py                 # logit lens, LatentLens, Patchscopes, SelfIE
  python run_unified_decoder.py --extended      # + tuned lens and embedding lens
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_object_identification import bootstrap_by_image  # noqa: E402
from run_failure_map import EXTRA_TOOLS, MAIN_TOOLS, SHORT, load_meta, load_scores  # noqa: E402
from vtr import evaluate as E  # noqa: E402
from vtr.geometry import HIGH_COVERAGE  # noqa: E402

RRF_K = 60


def log_softmax_t(s, temperature):
  z = s / temperature
  z = z - z.max(axis=1, keepdims=True)
  return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def fit_temperature(s, labels):
  """Temperature minimising the NLL of the true class (grid search, scale-aware)."""
  scale = np.nanstd(s) + 1e-8
  best_t, best_nll = scale, np.inf
  for t in scale * np.logspace(-2.5, 1.5, 41):
    nll = -log_softmax_t(s, t)[np.arange(len(labels)), labels].mean()
    if nll < best_nll:
      best_t, best_nll = t, nll
  return float(best_t)


def fit_weights(logps, labels, steps=200):
  """Non-negative per-tool weights maximising dev log-likelihood of sum_k w_k logp_k."""
  x = torch.tensor(np.stack(logps, 0), dtype=torch.float64)      # [K, n, C]
  y = torch.tensor(labels)
  raw = torch.zeros(x.shape[0], dtype=torch.float64, requires_grad=True)
  opt = torch.optim.LBFGS([raw], max_iter=steps, line_search_fn="strong_wolfe")

  def closure():
    opt.zero_grad()
    w = torch.nn.functional.softplus(raw)
    logits = (w[:, None, None] * x).sum(0)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    return loss

  opt.step(closure)
  return torch.nn.functional.softplus(raw).detach().numpy()


def split_images(meta, dev_fraction, seed):
  images = meta["image_id"].drop_duplicates().to_numpy()
  rng = np.random.RandomState(seed)
  dev = set(rng.choice(images, size=int(round(dev_fraction * len(images))), replace=False))
  return meta["image_id"].isin(dev).to_numpy()


def decode(scores, meta, class_names, tools, dev_fraction, seed):
  """Returns per-row correctness (rank) for every method and layer, plus fitted parameters."""
  index = {c: i for i, c in enumerate(class_names)}
  labels = meta["class_name"].map(index).to_numpy()
  layers = sorted(set.intersection(*[set(scores[t][0]) for t in tools]))
  dev = split_images(meta, dev_fraction, seed)
  fit_rows = dev & (meta["kind"] == "object").to_numpy() & \
      (meta["coverage"] >= HIGH_COVERAGE).to_numpy()

  scored = np.ones(len(meta), dtype=bool)
  for t in tools:
    ls, s = scores[t]
    scored &= ~np.isnan(s[[ls.index(l) for l in layers]]).all(axis=2).any(axis=0)
  fit_rows &= scored

  ranks, params = {}, []
  for l in layers:
    raw = {t: np.nan_to_num(scores[t][1][scores[t][0].index(l)], nan=0.0) for t in tools}
    temps = {t: fit_temperature(raw[t][fit_rows], labels[fit_rows]) for t in tools}
    logp = {t: log_softmax_t(raw[t], temps[t]) for t in tools}
    tool_ranks = {t: E.ranks(raw[t], labels) for t in tools}
    dev_acc = {t: float((tool_ranks[t][fit_rows] == 1).mean()) for t in tools}
    route = max(dev_acc, key=dev_acc.get)
    w = fit_weights([logp[t][fit_rows] for t in tools], labels[fit_rows])

    methods = dict(tool_ranks)
    methods["fusion_equal"] = E.ranks(np.mean([logp[t] for t in tools], 0), labels)
    methods["fusion_weighted"] = E.ranks(sum(wk * logp[t] for wk, t in zip(w, tools)), labels)
    # Reciprocal-rank sums tie often; break exact ties by the mean calibrated log-prob
    # (scaled far below one rank step) so pessimistic tie counting does not penalise RRF.
    rrf = sum(1.0 / (RRF_K + tool_ranks_all(raw[t])) for t in tools)
    tie_break = np.mean([logp[t] for t in tools], 0)
    tie_break = (tie_break - tie_break.min()) / (np.ptp(tie_break) + 1e-12) * 1e-9
    methods["rank_fusion"] = E.ranks(rrf + tie_break, labels)
    methods["routing"] = tool_ranks[route]
    ranks[l] = methods
    params.append({"layer": l, "route": route,
                   **{f"T_{t}": round(temps[t], 5) for t in tools},
                   **{f"w_{t}": round(float(wk), 4) for wk, t in zip(w, tools)},
                   **{f"dev_acc_{t}": round(dev_acc[t], 4) for t in tools}})
  return ranks, labels, dev, scored, layers, pd.DataFrame(params)


def tool_ranks_all(s):
  """Rank of every candidate (1 = best) per row; ties broken by order, for rank fusion."""
  order = np.argsort(-s, axis=1, kind="stable")
  r = np.empty_like(order)
  np.put_along_axis(r, order, np.arange(1, s.shape[1] + 1)[None, :].repeat(len(s), 0), axis=1)
  return r


def summarize(ranks, meta, dev, scored, layers, tools, n_boot, seed):
  test = ~dev & scored
  conds = {"object_high": (meta["kind"] == "object").to_numpy()
           & (meta["coverage"] >= HIGH_COVERAGE).to_numpy(),
           "pooled_mean": (meta["kind"] == "pooled_mean").to_numpy()}
  frames = []
  for cond, cmask in conds.items():
    mask = cmask & test
    if not mask.any():
      continue
    ids = meta["image_id"].to_numpy()[mask]
    for metric, k in (("top1", 1), ("top5", 5)):
      rows = []
      for l in layers:
        hit = {m: (r[mask] <= k).astype(float) for m, r in ranks[l].items()}
        acc = {t: hit[t].mean() for t in tools}
        best_t = max(acc, key=acc.get)
        hit["best_single_test"] = hit[best_t]
        hit["union"] = np.max([hit[t] for t in tools], axis=0)
        rows.append(pd.DataFrame({"image_id": ids, "layer": l, **hit}))
      long = pd.concat(rows, ignore_index=True)
      for m in [c for c in long.columns if c not in ("image_id", "layer")]:
        b = bootstrap_by_image(long, m, n_boot, seed)
        b["method"], b["condition"], b["metric"] = m, cond, metric
        frames.append(b)
  return pd.concat(frames, ignore_index=True)


def headline(summary, tools):
  rows = []
  for (cond, metric), g in summary.groupby(["condition", "metric"]):
    piv = g.pivot_table(index="layer", columns="method", values="mean")
    best, union = piv["best_single_test"], piv["union"]
    for m in piv.columns:
      gain = piv[m] - best
      possible = (union - best).replace(0, np.nan)
      rows.append({"condition": cond, "metric": metric, "method": m,
                   "mean_acc_over_layers": round(float(piv[m].mean()), 4),
                   "mean_gain_vs_best_single": round(float(gain.mean()), 4),
                   "layers_at_or_above_best_single": int((gain >= -0.005).sum()),
                   "n_layers": len(piv),
                   "share_of_possible_gain": round(float((gain / possible).mean()), 3)})
  return pd.DataFrame(rows)


PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


def plot(summary, tools, path):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  s = summary[(summary.condition == "object_high") & (summary.metric == "top1")]
  piv = s.pivot_table(index="layer", columns="method", values="mean")
  lo = s.pivot_table(index="layer", columns="method", values="ci_lo")
  hi = s.pivot_table(index="layer", columns="method", values="ci_hi")
  fig, ax = plt.subplots(figsize=(10.5, 5.5))
  for i, t in enumerate(tools):
    ax.plot(piv.index, piv[t], color=PALETTE[i % len(PALETTE)], lw=1.3, alpha=0.8,
            label=SHORT[t])
  ax.plot(piv.index, piv["fusion_weighted"], color="#17191c", lw=2.6, label="unified (weighted fusion)")
  ax.fill_between(piv.index, lo["fusion_weighted"], hi["fusion_weighted"], color="#17191c",
                  alpha=0.12, lw=0)
  ax.plot(piv.index, piv["rank_fusion"], color="#17191c", lw=1.4, ls="-.", label="rank fusion")
  ax.plot(piv.index, piv["union"], color="#9a9994", lw=1.5, ls="--", label="union (upper bound)")
  ax.set_xlabel("LLM layer")
  ax.set_ylabel("top-1 accuracy, held-out test images")
  ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
  ax.set_title("Unified decoder vs single readouts (object tokens)", loc="left", fontsize=11)
  for side in ("top", "right"):
    ax.spines[side].set_visible(False)
  ax.grid(axis="y", alpha=0.3)
  ax.legend(frameon=False, fontsize=8, ncol=2)
  fig.tight_layout()
  fig.savefig(path, dpi=150)
  plt.close(fig)


def parse_args(argv=None):
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--vtr_dir", default="./results/vtr_llava15")
  p.add_argument("--out_dir", default=None, help="default: <vtr_dir>/unified[_extended]")
  p.add_argument("--extended", action="store_true", help="add tuned lens and embedding lens")
  p.add_argument("--all_classes", action="store_true")
  p.add_argument("--dev_fraction", type=float, default=0.3)
  p.add_argument("--n_boot", type=int, default=1000)
  p.add_argument("--seed", type=int, default=0)
  return p.parse_args(argv)


def main(argv=None):
  args = parse_args(argv)
  out_dir = args.out_dir or os.path.join(args.vtr_dir,
                                         "unified" + ("_extended" if args.extended else ""))
  os.makedirs(out_dir, exist_ok=True)
  with open(os.path.join(args.vtr_dir, "samples.json"), encoding="utf-8") as f:
    class_names = json.load(f)["class_names"]
  meta = load_meta(args.vtr_dir)
  wanted = MAIN_TOOLS + (EXTRA_TOOLS if args.extended else [])
  scores = load_scores(args.vtr_dir, wanted)
  tools = [t for t in wanted if t in scores]
  if not args.all_classes:
    with open(os.path.join(args.vtr_dir, "fair_classes.json"), encoding="utf-8") as f:
      kept = json.load(f)["kept"]
    scores, meta, class_names = E.subset(scores, meta, class_names,
                                         [class_names.index(c) for c in kept])

  ranks, labels, dev, scored, layers, params = decode(scores, meta, class_names, tools,
                                                      args.dev_fraction, args.seed)
  print(f"tools: {tools}; {len(class_names)} classes; "
        f"{meta.loc[dev, 'image_id'].nunique()} dev / {meta.loc[~dev, 'image_id'].nunique()} "
        f"test images; layers {layers}")
  params.to_csv(os.path.join(out_dir, "unified_weights.csv"), index=False)
  summary = summarize(ranks, meta, dev, scored, layers, tools, args.n_boot, args.seed)
  summary.to_csv(os.path.join(out_dir, "unified_summary.csv"), index=False)
  head = headline(summary, tools)
  head.to_csv(os.path.join(out_dir, "unified_headline.csv"), index=False)
  plot(summary, tools, os.path.join(out_dir, "unified_by_layer.png"))

  pd.set_option("display.width", 250)
  view = summary[(summary.condition == "object_high") & (summary.metric == "top1")]
  table = view.pivot_table(index="layer", columns="method", values="mean").round(3)
  order = tools + ["best_single_test", "routing", "fusion_equal", "fusion_weighted",
                   "rank_fusion", "union"]
  print("\nTest images, object tokens, top-1:")
  print(table[order].rename(columns=lambda c: SHORT.get(c, c)).to_string())
  h = head[(head.condition == "object_high") & (head.metric == "top1")].set_index("method")
  print("\nAggregate (mean over layers), object tokens, top-1:")
  print(h.loc[order, ["mean_acc_over_layers", "mean_gain_vs_best_single",
                      "layers_at_or_above_best_single", "share_of_possible_gain"]].to_string())
  print("\nRouting choice and fusion weights per layer:")
  print(params[["layer", "route"] + [c for c in params.columns if c.startswith("w_")]]
        .to_string(index=False))
  print(f"\nAll outputs in {out_dir}")


if __name__ == "__main__":
  main()
