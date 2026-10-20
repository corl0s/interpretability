r"""Prototype: injection variants designed for visual tokens (LLaVA), on the sweep's rows.

Same rows as run_ps_sweep.py (200 images: object tokens with coverage >= 0.5, the mean-pooled
object state, the norm-matched random control). For every variant, source layer and target layer
the state is injected and the classes are scored first-token closed-set (vtr/injection.py).

Variants (see vtr/injection.py):
  selfie_text          SelfIE text prompt (baseline, same first-token scoring)
  selfie_text_norm       + norm matching
  img_all              neutral-image prompt, state at every image token
  img_all_norm           + norm matching
  img_all_multi          + re-injection at target + 4 and + 8
  img_own              state at the token's own grid position
  img_own3x3           state at the 3x3 block around it
  img_footprint        state at every token the object covers
  img_footprint_norm     + norm matching
  img_all_rewind       img_all, but the layer-s state is first translated to layer t's format by
                       a label-free rewind map (fit_rewind_maps.py, vtr/rewind.py)
  selfie_text_rewind   selfie_text with the same rewind map (does the stage matter in text slots?)

Selection is honest: images are split into development (30%) and test (70%); the target layer
of each (variant, source layer) is chosen on development images and accuracy is reported on test
images, with image-clustered CIs. Main numbers use the fair class subset.

Outputs in --out_dir:
  cells/<variant>/s{S}_t{T}.npz   raw and PMI scores
  proto_summary.csv              accuracy per variant / source / target / condition / split
  proto_best.csv                 per variant and source: dev-chosen target, test accuracy
  proto_by_source.png

  python run_injection_proto.py                                   # all variants (~2-3 h)
  python run_injection_proto.py --variants selfie_text img_all     # a subset
  python run_injection_proto.py --summarize_only                   # tables from finished cells

Rewind variants need <vtr_dir>/rewind/maps.pt (fit_rewind_maps.py); without it they are skipped.
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
from run_ps_sweep import select_rows  # noqa: E402
from vtr import evaluate as E  # noqa: E402
from vtr.injection import IMAGE_QUESTION, InjectionReadout  # noqa: E402
from vtr.readouts import class_forms  # noqa: E402
from vtr.rewind import RewindMaps  # noqa: E402

VARIANTS = {
    "selfie_text": dict(target="text", slots="placeholders"),
    "selfie_text_norm": dict(target="text", slots="placeholders", norm_match=True),
    "img_all": dict(target="image", slots="all"),
    "img_all_norm": dict(target="image", slots="all", norm_match=True),
    "img_all_multi": dict(target="image", slots="all", extra_layers=(4, 8)),
    "img_own": dict(target="image", slots="own"),
    "img_own3x3": dict(target="image", slots="own3x3"),
    "img_footprint": dict(target="image", slots="footprint"),
    "img_footprint_norm": dict(target="image", slots="footprint", norm_match=True),
    "img_all_rewind": dict(target="image", slots="all", rewind=True),
    "selfie_text_rewind": dict(target="text", slots="placeholders", rewind=True),
}
CONDITIONS = ("object", "pooled_mean", "random")


def parse_args(argv=None):
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--vtr_dir", default="./results/vtr_llava15")
  p.add_argument("--out_dir", default="./results/vtr_llava15/injection_proto")
  p.add_argument("--model_name", default="llava-hf/llava-1.5-7b-hf")
  p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
  p.add_argument("--sources", default="0,4,8,12,16,20,24,28")
  p.add_argument("--targets", default="0,4,8,12,16")
  p.add_argument("--n_images", type=int, default=200)
  p.add_argument("--question", default=IMAGE_QUESTION)
  p.add_argument("--rows_per_batch", type=int, default=16)
  p.add_argument("--dev_fraction", type=float, default=0.3)
  p.add_argument("--n_boot", type=int, default=500)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  p.add_argument("--rewind_maps", default=None, help="default: <vtr_dir>/rewind/maps.pt")
  p.add_argument("--summarize_only", action="store_true",
                 help="no model: summarize the variants whose cells are all finished")
  return p.parse_args(argv)


def run_cells(args, mt, states, meta_rows, rows, coverage, class_names, sources, targets,
              maps=None):
  forms = class_forms(class_names, synonyms=False)
  readouts = {}
  for name in args.variants:
    v = VARIANTS[name]
    if v["target"] not in readouts:
      readouts[v["target"]] = InjectionReadout(mt, class_names, forms, v["target"], args.device,
                                               args.question, args.rows_per_batch)
    ro = readouts[v["target"]]
    positions = [ro.slots(v["slots"], int(r.token), coverage[r.image_id])
                 for r in meta_rows.itertuples()]
    cell_dir = os.path.join(args.out_dir, "cells", name)
    os.makedirs(cell_dir, exist_ok=True)
    for s in sources:
      h = states[rows, s].float()
      for t in targets:
        path = os.path.join(cell_dir, f"s{s}_t{t}.npz")
        if os.path.exists(path):
          continue
        t0 = time.time()
        h_in = h
        if v.get("rewind"):
          if not maps.has(s, t):
            raise KeyError(f"no rewind map for {s} -> {t}; refit with these sources/targets")
          h_in = maps.apply(h, s, t)
        raw, pmi = ro.scores(h_in, t, positions, v.get("norm_match", False),
                             v.get("extra_layers", ()))
        np.savez(path, raw=raw.astype(np.float16), pmi=pmi.astype(np.float16))
        print(f"  {name}: source {s:2d} -> target {t:2d} ({time.time() - t0:.1f}s)", flush=True)
  return readouts


def shared_first_tokens(readouts, class_names, keep):
  """Pairs of kept classes whose forms share a first token (first-token scoring ties them)."""
  pairs = set()
  for ro in readouts.values():
    owner, first = ro.owner.tolist(), ro.first.tolist()
    by_tok = {}
    for o, f in zip(owner, first):
      if o in keep:
        by_tok.setdefault(f, set()).add(class_names[o])
    pairs |= {tuple(sorted(c)) for c in by_tok.values() if len(c) > 1}
  return sorted(pairs)


def summarize(args, meta_rows, class_names, sources, targets, keep):
  index = {c: i for i, c in enumerate(class_names)}
  labels = meta_rows["class_name"].map(index).to_numpy()
  ok = np.isin(labels, keep)
  remap = {c: i for i, c in enumerate(keep)}
  lab = np.array([remap[x] for x in labels[ok]])
  m = meta_rows[ok].reset_index(drop=True)
  images = m["image_id"].drop_duplicates().sample(frac=1.0, random_state=args.seed).to_numpy()
  dev_images = set(images[:int(round(args.dev_fraction * len(images)))])
  split = np.where(m["image_id"].isin(dev_images), "dev", "test")

  frames = []
  for name in args.variants:
    for s in sources:
      for t in targets:
        cell = np.load(os.path.join(args.out_dir, "cells", name, f"s{s}_t{t}.npz"))
        for variant in ("pmi", "raw"):
          rank = E.ranks(cell[variant].astype(np.float32)[ok][:, keep], lab)
          for cond in CONDITIONS:
            for part in ("dev", "test"):
              mask = ((m["kind"] == cond).to_numpy()) & (split == part)
              if not mask.any():
                continue
              for metric, k in (("top1", 1), ("top5", 5)):
                df = pd.DataFrame({"image_id": m["image_id"].to_numpy()[mask], "layer": 0,
                                   "score": (rank[mask] <= k).astype(float)})
                b = bootstrap_by_image(df, "score", args.n_boot, args.seed).iloc[0]
                frames.append({"variant": name, "source": s, "target": t, "scoring": variant,
                               "condition": cond, "split": part, "metric": metric,
                               "mean": b["mean"], "ci_lo": b["ci_lo"], "ci_hi": b["ci_hi"],
                               "n_rows": int(mask.sum())})
  out = pd.DataFrame(frames)
  out.to_csv(os.path.join(args.out_dir, "proto_summary.csv"), index=False)

  key = ["variant", "source", "target"]
  sel = out[(out.scoring == "pmi") & (out.metric == "top1")]
  dev = sel[(sel.condition == "object") & (sel.split == "dev")]
  test = sel[sel.split == "test"].pivot_table(index=key, columns="condition", values="mean")
  best = []
  for (name, s), g in dev.groupby(["variant", "source"]):
    t = int(g.loc[g["mean"].idxmax(), "target"])
    row = test.loc[(name, s, t)]
    ci = sel[(sel.variant == name) & (sel.source == s) & (sel.target == t)
             & (sel.condition == "object") & (sel.split == "test")].iloc[0]
    best.append({"variant": name, "source": s, "target_chosen_on_dev": t,
                 "test_object": row["object"], "test_ci_lo": ci["ci_lo"],
                 "test_ci_hi": ci["ci_hi"], "test_pooled_mean": row.get("pooled_mean"),
                 "test_random": row.get("random")})
  best = pd.DataFrame(best)
  best.to_csv(os.path.join(args.out_dir, "proto_best.csv"), index=False)
  return best


def probe_reference(vtr_dir, sources):
  path = os.path.join(vtr_dir, "summary_fair.csv")
  if not os.path.exists(path):
    return None
  s = pd.read_csv(path)
  s = s[(s.readout == "probe_linear") & (s.condition == "object_high") & (s.metric == "top1")]
  return s.set_index("layer")["mean"].reindex(sources)


def plot(best, probe, path, n_classes):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  palette = E.PALETTE + ["#6b6b6b"]
  fig, ax = plt.subplots(figsize=(8, 4.6))
  for i, (name, g) in enumerate(best.groupby("variant", sort=False)):
    ax.plot(g.source, g.test_object, "-o", ms=3, lw=2 if name in ("selfie_text", "img_all")
            else 1.3, color=palette[i % len(palette)], label=name)
  if probe is not None:
    ax.plot(probe.index, probe.values, ":", color="#17191c", lw=1.5, label="probe (reference)")
  ax.axhline(1 / n_classes, color="#9a9994", lw=1, ls=":")
  ax.set_xlabel("source layer")
  ax.set_ylabel("top-1, object tokens (test images)")
  ax.set_title("Injection variants, target layer chosen on development images", loc="left",
               fontsize=10)
  ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
  ax.grid(axis="y", alpha=0.3)
  for side in ("top", "right"):
    ax.spines[side].set_visible(False)
  ax.legend(frameon=False, fontsize=8, ncol=2)
  fig.tight_layout()
  fig.savefig(path, dpi=150)
  plt.close(fig)


def main(argv=None, mt=None):
  args = parse_args(argv)
  os.makedirs(args.out_dir, exist_ok=True)
  sources = [int(x) for x in args.sources.split(",")]
  targets = [int(x) for x in args.targets.split(",")]
  with open(os.path.join(args.vtr_dir, "samples.json"), encoding="utf-8") as f:
    data = json.load(f)
  class_names = data["class_names"]
  coverage = {s["image_id"]: s["coverage"] for s in data["samples"]}
  cache = torch.load(os.path.join(args.vtr_dir, "states.pt"), weights_only=False)
  states, meta = cache["states"], cache["meta"]
  rows = select_rows(meta, args.n_images, args.seed)
  meta_rows = meta.iloc[rows].reset_index(drop=True)
  print(f"{meta_rows['image_id'].nunique()} images, {len(rows)} rows "
        f"({meta_rows['kind'].value_counts().to_dict()}); variants {args.variants}; "
        f"sources {sources} x targets {targets}")
  if args.summarize_only:
    args.variants = [v for v in args.variants if all(
        os.path.exists(os.path.join(args.out_dir, "cells", v, f"s{s}_t{t}.npz"))
        for s in sources for t in targets)]
    print(f"summarizing finished variants only: {args.variants}")
    if not args.variants:
      return
    readouts = {}
  else:
    maps = None
    if any(VARIANTS[v].get("rewind") for v in args.variants):
      path = args.rewind_maps or os.path.join(args.vtr_dir, "rewind", "maps.pt")
      if os.path.exists(path):
        maps = RewindMaps(path)
        print(f"rewind maps from {path} ({len(maps.maps)} pairs)")
      else:
        args.variants = [v for v in args.variants if not VARIANTS[v].get("rewind")]
        print(f"no rewind maps at {path}: skipping the rewind variants")
    with open(os.path.join(args.out_dir, "config.json"), "w", encoding="utf-8") as f:
      json.dump({**vars(args), "rows": rows.tolist()}, f)
    if mt is None:
      from general_utils import ModelAndTokenizer
      dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
      mt = ModelAndTokenizer(args.model_name, torch_dtype=dtype, device=args.device)
    readouts = run_cells(args, mt, states, meta_rows, rows, coverage, class_names, sources,
                         targets, maps)

  fair_path = os.path.join(args.vtr_dir, "fair_classes.json")
  if os.path.exists(fair_path):
    with open(fair_path, encoding="utf-8") as f:
      keep = [class_names.index(c) for c in json.load(f)["kept"]]
  else:
    keep = list(range(len(class_names)))
  ties = shared_first_tokens(readouts, class_names, keep)
  print(f"\n{len(keep)} classes; kept classes sharing a first token: {ties or 'none'}")

  best = summarize(args, meta_rows, class_names, sources, targets, keep)
  probe = probe_reference(args.vtr_dir, sources)
  table = best.pivot(index="variant", columns="source", values="test_object").reindex(
      args.variants)
  if probe is not None:
    table.loc["probe (reference)"] = probe.values
  table["mean"] = table.mean(axis=1)
  pd.set_option("display.width", 200)
  print("\nTop-1, object tokens, test images (target layer chosen on development images):")
  print(table.round(3).to_string())
  print("\nControls at the chosen target (random vector, test images):")
  print(best.pivot(index="variant", columns="source", values="test_random")
        .reindex(args.variants).round(3).to_string())
  print("\nTarget layer chosen on development images:")
  print(best.pivot(index="variant", columns="source", values="target_chosen_on_dev")
        .reindex(args.variants).to_string())
  plot(best, probe, os.path.join(args.out_dir, "proto_by_source.png"), len(keep))
  print(f"\nAll outputs in {args.out_dir}")


if __name__ == "__main__":
  main()
