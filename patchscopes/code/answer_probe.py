r"""Same-data control for the visual activation oracle: a linear probe trained on the oracle's own
training data (train_oracle.py), so the only difference left is the reader.

The oracle learns to answer from a state injected into image slots, with LLaVA's own answer
about the token's crop as the target. Here the same tokens and answers train a linear classifier
instead. A probe needs a fixed label set, so each answer is mapped to the fair benchmark class it
names (vtr.readouts.class_forms with synonyms, as train_oracle.py does to drop held-out classes),
and answers naming no fair class become one "other" label. Unusable answers and answers naming a
held-out class are dropped exactly as for the oracle, so the probe never sees a held-out class and
cannot output one: its held-out score is 0 by construction.

Two probes, both StandardScaler + logistic regression like the benchmark's probe_linear:
  answer_probe         one per source layer, on every kept token at that layer
  answer_probe_pooled  one for all layers, on exactly the oracle's (token, layer) training pairs

Scored like train_oracle.py's eval: object tokens of the 200 benchmark images (select_rows), test
images only (the same 30/70 dev/test split), fair classes, closed-set top-1; then split into
seen / held-out classes and easy / hard tokens. The oracle and the untrained image-slot readout
are read from the oracle's eval cells, and the COCO-label probe from scores/probe_linear.npz.

  python answer_probe.py --vtr_dir ./results/vtr_llava15 \
      --oracle_dir ./results/vtr_llava15/oracle_crop7_add
    -> <oracle_dir>/answer_probe/{scores.npz, table.csv, groups.csv}
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_vtr  # noqa: E402
from run_ps_sweep import select_rows  # noqa: E402
from train_oracle import load_benchmark, load_data, names_class, training_pool  # noqa: E402
from vtr import evaluate as E  # noqa: E402
from vtr.readouts import class_forms  # noqa: E402

OTHER = "<other>"


def answer_labels(answers, pool, kept, held, class_names):
  """Fair class named by each pooled answer (first match in `kept` order), else OTHER."""
  forms = class_forms(class_names, synonyms=True)
  seen = [c for c in kept if c not in held]
  out = []
  for i in pool:
    a = answers[i]
    out.append(next((c for c in seen if names_class(a, forms[c])), OTHER))
  return np.array(out)


def fit(X, y, seed):
  from sklearn.linear_model import LogisticRegression
  from sklearn.pipeline import make_pipeline
  from sklearn.preprocessing import StandardScaler
  clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
  clf.fit(X, y)
  return clf


def class_logp(clf, X, class_names):
  """[n, n_classes] log-probabilities; classes the probe never saw (held-out) stay -inf."""
  proba = clf.predict_proba(X)
  out = np.full((len(X), len(class_names)), -np.inf, dtype=np.float32)
  index = {c: i for i, c in enumerate(class_names)}
  for j, c in enumerate(clf.classes_):
    if c in index:
      out[:, index[c]] = np.log(proba[:, j] + 1e-12)
  return out


def main(argv=None):
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--vtr_dir", default="./results/vtr_llava15")
  p.add_argument("--oracle_dir", default="./results/vtr_llava15/oracle_crop7_add")
  p.add_argument("--data_dir", default=None, help="default: the oracle's own training shards")
  args = p.parse_args(argv)

  blob = torch.load(os.path.join(args.oracle_dir, "lora.pt"), weights_only=False,
                    map_location="cpu")
  cfg = blob["config"]
  del blob
  seed, t = cfg["seed"], cfg["target_layer"]
  data_dir = args.data_dir or cfg.get("data_dir") or os.path.join(args.oracle_dir, "data")
  bench, kept = load_benchmark(args.vtr_dir)
  class_names = bench["class_names"]
  with open(os.path.join(args.oracle_dir, "holdout.json"), encoding="utf-8") as f:
    held = json.load(f)
  out_dir = os.path.join(args.oracle_dir, "answer_probe")
  os.makedirs(out_dir, exist_ok=True)

  # ---- training data: the oracle's pool, labelled by the class each answer names
  data = load_data(data_dir)
  layers = list(data["layers"])
  pool = training_pool(SimpleArgs(), data, held, class_names)
  y = answer_labels(data["answers"], pool, kept, held, class_names)
  counts = pd.Series(y).value_counts()
  print(f"  {len(pool)} tokens; {(y != OTHER).sum()} name a seen fair class "
        f"({(counts.index != OTHER).sum()} classes), {(y == OTHER).sum()} -> {OTHER}")
  n_layers = len(layers)
  pairs = np.stack(np.meshgrid(np.arange(len(pool)), np.arange(n_layers), indexing="ij"),
                   -1).reshape(-1, 2)
  pairs = pairs[np.random.RandomState(seed).permutation(len(pairs))][:cfg["train_examples"]]

  # ---- evaluation rows: as train_oracle.py's eval
  cache = torch.load(os.path.join(args.vtr_dir, "states.pt"), weights_only=False)
  states, meta = cache["states"], cache["meta"]
  rows = select_rows(meta, cfg["eval_images"], seed)
  meta_rows = meta.iloc[rows].reset_index(drop=True)

  scores_path = os.path.join(out_dir, "scores.npz")
  if os.path.exists(scores_path):
    saved = np.load(scores_path)
    scores = {k: saved[k] for k in saved.files}
  else:
    scores = {}
    Xs = data["states"].float().numpy()
    for li, s in enumerate(layers):
      clf = fit(Xs[pool, li], y, seed)
      scores[f"answer_probe_s{s}"] = class_logp(clf, states[rows, s].float().numpy(), class_names)
      print(f"  answer_probe layer {s}: trained on {len(pool)} tokens", flush=True)
    clf = fit(Xs[pool[pairs[:, 0]], pairs[:, 1]], y[pairs[:, 0]], seed)
    for s in layers:
      scores[f"answer_probe_pooled_s{s}"] = class_logp(clf, states[rows, s].float().numpy(),
                                                       class_names)
    print(f"  answer_probe_pooled: trained on the oracle's {len(pairs)} (token, layer) pairs")
    np.savez(scores_path, **scores)

  table, groups = compare(args, cfg, scores, rows, meta_rows, class_names, kept, held, layers, t)
  table.to_csv(os.path.join(out_dir, "table.csv"))
  groups.to_csv(os.path.join(out_dir, "groups.csv"))
  pd.set_option("display.width", 200)
  print(f"\nTop-1 (%), object tokens of test images, {len(kept)} fair classes, by source layer:")
  print((table * 100).round(1).to_string())
  print("\nMean over source layers (%), by group:")
  n = groups.loc["(test object rows)"].astype(int)
  print((groups.drop(index="(test object rows)") * 100).round(1).to_string())
  print("test object rows: " + ", ".join(f"{g} {k}" for g, k in n.items()))
  print(f"\nAll outputs in {out_dir}")


class SimpleArgs:
  """training_pool only reads its data and class lists; it takes args for its signature."""


def compare(args, cfg, scores, rows, meta_rows, class_names, kept, held, layers, t):
  index = {c: i for i, c in enumerate(class_names)}
  keep = [index[c] for c in kept]
  labels = meta_rows["class_name"].map(index).to_numpy()
  ok = np.isin(labels, keep)
  remap = {c: i for i, c in enumerate(keep)}
  m = meta_rows[ok].reset_index(drop=True)
  lab = np.array([remap[x] for x in labels[ok]])
  images = m["image_id"].drop_duplicates().sample(frac=1.0, random_state=cfg["seed"]).to_numpy()
  dev = set(images[:int(round(0.3 * len(images)))])
  test_obj = (~m["image_id"].isin(dev) & (m["kind"] == "object")).to_numpy()
  is_held = m["class_name"].isin(held).to_numpy()
  main_path = os.path.join(args.oracle_dir, "eval", "main_object.csv")
  main = None
  if os.path.exists(main_path):
    df = pd.read_csv(main_path, keep_default_na=False)
    main = dict(zip(df["image_id"], df["main_object"]))
  is_hard = (m["class_name"] != m["image_id"].map(main)).to_numpy() if main else None

  def hits(cell):
    return E.ranks(cell.astype(np.float32)[ok][:, keep], lab) == 1

  readouts = {}
  for name in ("oracle", "untrained"):
    for s in layers:
      path = os.path.join(args.oracle_dir, "eval", "cells", name, f"s{s}_t{t}.npz")
      if os.path.exists(path):
        readouts[("oracle" if name == "oracle" else "image_slot (untrained)", s)] = hits(
            np.load(path)["pmi"])
  for s in layers:
    readouts[("answer_probe", s)] = hits(scores[f"answer_probe_s{s}"])
    readouts[("answer_probe_pooled", s)] = hits(scores[f"answer_probe_pooled_s{s}"])
  probe_path = os.path.join(args.vtr_dir, "scores", "probe_linear.npz")
  if os.path.exists(probe_path):
    pl, ps = run_vtr.load_scores(probe_path)
    for s in layers:
      if s in pl:
        readouts[("probe_linear (COCO labels)", s)] = hits(ps[pl.index(s)][rows])

  masks = {"all": test_obj, "seen": test_obj & ~is_held, "held-out": test_obj & is_held}
  if is_hard is not None:
    masks.update({"easy": test_obj & ~is_hard, "hard": test_obj & is_hard})
  long = pd.DataFrame([{"readout": r, "source": s, "group": g,
                        "top1": h[mask].mean() if mask.any() else np.nan}
                       for (r, s), h in readouts.items() for g, mask in masks.items()])
  table = long[long.group == "all"].pivot(index="readout", columns="source", values="top1")
  table.insert(0, "mean", table.mean(axis=1))
  groups = long.groupby(["readout", "group"])["top1"].mean().unstack()[list(masks)]
  groups.loc["(test object rows)"] = [int(mask.sum()) for mask in masks.values()]
  return table, groups


if __name__ == "__main__":
  main()
