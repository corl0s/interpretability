r"""COCO-VTR Milestone 1: every readout on LLaVA-1.5-7B, object-identity task, closed-set.

Stages (each checkpointed in --out_dir; re-runs skip finished stages unless --overwrite):
  data      select COCO val2017 images / objects, per-token coverage    -> samples.json
  cache     hidden states of sampled tokens + pooled + random, all layers -> states.pt
  readouts  scores over the 80 COCO classes for each readout and layer  -> scores/*.npz
  eval      accuracies per layer / condition, image-clustered CIs, plots -> summary.csv ...
            plus the same on the "fair" class subset (every training-free readout can
            score and tell the classes apart)                        -> *_fair.*, fair_classes.json

Readout names in the outputs:
  logit_lens, logit_lens_syn         class name only / with unambiguous synonyms
  latentlens, latentlens_syn         same two variants (needs a LatentLens bank for LLaVA)
  patchscopes_raw, patchscopes_pmi   continuation log-likelihood; PMI subtracts the prompt prior
  probe_linear, probe_mlp            supervised references (availability)

Usage on SCC (one GPU):
  # smoke run
  python run_vtr.py --coco_dir /projectnb/mlresearch/vishnuav/coco --out_dir ./results/vtr_smoke \
      --n_images 20 --layers 0,2,8,16,24,31 --n_boot 200
  # full run
  python run_vtr.py --coco_dir /projectnb/mlresearch/vishnuav/coco
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

from vtr import evaluate as E  # noqa: E402
from vtr.cache import cache_states  # noqa: E402
from vtr.data import select_object_samples, summarize_samples  # noqa: E402
from vtr.geometry import GEOMETRIES  # noqa: E402
from vtr.readouts import (LatentLens, LogitLens, Patchscopes, class_forms,  # noqa: E402
                          probe_scores)

MAIN_READOUTS = ["logit_lens", "latentlens", "patchscopes_pmi", "probe_linear"]
# Readouts whose class coverage defines the fair subset (the probe is a supervised reference).
# patchscopes_pmi is the main Patchscopes number, fixed before looking at results.
TRAINING_FREE = ["logit_lens", "latentlens", "patchscopes_pmi"]


def parse_args(argv=None):
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--stage", default="all", choices=["all", "data", "cache", "readouts", "eval"])
  p.add_argument("--coco_dir", default=None)
  p.add_argument("--out_dir", default="./results/vtr_llava15")
  p.add_argument("--model_name", default="llava-hf/llava-1.5-7b-hf")
  p.add_argument("--geometry", default="llava-1.5", choices=sorted(GEOMETRIES))
  # data
  p.add_argument("--n_images", type=int, default=1000)
  p.add_argument("--per_class", type=int, default=20, help="max images per class")
  p.add_argument("--min_area", type=float, default=4000)
  p.add_argument("--max_area", type=float, default=0, help="0 = no upper bound")
  p.add_argument("--min_tokens", type=int, default=4, help="tokens with coverage >= 0.5")
  # cache
  p.add_argument("--per_bin", type=int, default=2, help="object tokens per overlap bin")
  p.add_argument("--n_outside", type=int, default=2)
  # readouts
  p.add_argument("--readouts", nargs="+", default=["logit_lens", "latentlens", "patchscopes",
                                                   "probe"])
  p.add_argument("--layers", default="all", help="'all' or comma-separated layer indices")
  p.add_argument("--latentlens_bank", default="./results/latentlens_index/bank")
  p.add_argument("--ps_prompt", default="identity", choices=["identity", "entity"])
  p.add_argument("--ps_synonyms", action="store_true",
                 help="also score synonyms in Patchscopes (about 2x slower)")
  p.add_argument("--ps_max_images", type=int, default=0, help="0 = all images")
  p.add_argument("--ps_rows_per_batch", type=int, default=256)
  p.add_argument("--probe_splits", type=int, default=5)
  # eval
  p.add_argument("--n_boot", type=int, default=2000)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  p.add_argument("--overwrite", action="store_true")
  return p.parse_args(argv)


def load_model(args):
  from general_utils import ModelAndTokenizer
  dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
  print(f"Loading {args.model_name} ...")
  return ModelAndTokenizer(args.model_name, torch_dtype=dtype, device=args.device)


def save_scores(path, layers, scores):
  np.savez(path, layers=np.asarray(layers), scores=np.asarray(scores, dtype=np.float16))


def load_scores(path):
  d = np.load(path)
  return list(d["layers"]), d["scores"].astype(np.float32)


def run_readouts(args, mt, states, meta, class_names, layers):
  score_dir = os.path.join(args.out_dir, "scores")
  os.makedirs(score_dir, exist_ok=True)

  def done(*names):
    return not args.overwrite and all(
        os.path.exists(os.path.join(score_dir, f"{n}.npz")) for n in names)

  def per_layer(fn):
    out = []
    for l in layers:
      t0 = time.time()
      out.append(fn(states[:, l].float(), l))
      print(f"    layer {l} ({time.time() - t0:.1f}s)")
    return out

  plain = class_forms(class_names, synonyms=False)
  syn = class_forms(class_names, synonyms=True)

  if "logit_lens" in args.readouts:
    for name, forms in (("logit_lens", plain), ("logit_lens_syn", syn)):
      if done(name):
        continue
      print(f"Readout: {name}")
      ro = LogitLens(mt, class_names, forms, args.device)
      save_scores(os.path.join(score_dir, f"{name}.npz"), layers, per_layer(ro.scores))

  if "latentlens" in args.readouts:
    for name, forms in (("latentlens", plain), ("latentlens_syn", syn)):
      if done(name):
        continue
      print(f"Readout: {name} (bank: {args.latentlens_bank})")
      ro = LatentLens(args.latentlens_bank, mt.tokenizer, class_names, forms, args.device,
                      cache_file=os.path.join(score_dir, f"{name}_entries.npz"))
      save_scores(os.path.join(score_dir, f"{name}.npz"), layers, per_layer(ro.scores))
      del ro

  if "patchscopes" in args.readouts and not done("patchscopes_raw", "patchscopes_pmi"):
    print(f"Readout: patchscopes ({args.ps_prompt} prompt)")
    ro = Patchscopes(mt, class_names, syn if args.ps_synonyms else plain, args.ps_prompt,
                     args.device, rows_per_batch=args.ps_rows_per_batch)
    rows = np.arange(len(meta))
    if args.ps_max_images:
      keep = meta["image_id"].drop_duplicates().iloc[:args.ps_max_images]
      rows = np.flatnonzero(meta["image_id"].isin(keep).to_numpy())
    raw = np.full((len(layers), len(meta), len(class_names)), np.nan, dtype=np.float32)
    pmi = raw.copy()
    for i, l in enumerate(layers):
      t0 = time.time()
      r, p = ro.scores(states[rows, l].float(), l)
      raw[i, rows], pmi[i, rows] = r, p
      print(f"    layer {l}: {len(rows)} rows ({time.time() - t0:.1f}s)")
    save_scores(os.path.join(score_dir, "patchscopes_raw.npz"), layers, raw)
    save_scores(os.path.join(score_dir, "patchscopes_pmi.npz"), layers, pmi)

  if "probe" in args.readouts:
    for kind in ("linear", "mlp"):
      name = f"probe_{kind}"
      if done(name):
        continue
      print(f"Readout: {name}")
      out = per_layer(lambda h, l: probe_scores(h.numpy(), meta, class_names, kind,
                                                args.probe_splits, args.seed))
      save_scores(os.path.join(score_dir, f"{name}.npz"), layers, out)


def main(argv=None):
  args = parse_args(argv)
  os.makedirs(args.out_dir, exist_ok=True)
  geometry = GEOMETRIES[args.geometry]()
  stages = ["data", "cache", "readouts", "eval"] if args.stage == "all" else [args.stage]

  # ------------------------------------------------------------------ data
  samples_path = os.path.join(args.out_dir, "samples.json")
  if "data" in stages and (args.overwrite or not os.path.exists(samples_path)):
    from visual_object_identification import ensure_annotations
    print("Selecting COCO-VTR object samples ...")
    samples, class_names = select_object_samples(
        ensure_annotations(args.coco_dir), geometry, args.n_images, args.per_class,
        args.min_area, args.max_area or None, args.min_tokens, args.seed)
    with open(samples_path, "w", encoding="utf-8") as f:
      json.dump({"class_names": class_names, "geometry": geometry.name, "samples": samples}, f)
    summary = summarize_samples(samples)
    with open(os.path.join(args.out_dir, "data_summary.json"), "w", encoding="utf-8") as f:
      json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
  with open(samples_path, "r", encoding="utf-8") as f:
    data = json.load(f)
  samples, class_names = data["samples"], data["class_names"]

  # ------------------------------------------------------------------ cache
  mt = None
  cache_path = os.path.join(args.out_dir, "states.pt")
  if "cache" in stages and (args.overwrite or not os.path.exists(cache_path)):
    mt = load_model(args)
    print(f"Caching hidden states for {len(samples)} images ...")
    states, meta = cache_states(mt, mt.processor, samples, args.coco_dir, args.per_bin,
                                args.n_outside, args.seed)
    torch.save({"states": states, "meta": meta}, cache_path)
  if not os.path.exists(cache_path):
    return
  cache = torch.load(cache_path, weights_only=False)
  states, meta = cache["states"], cache["meta"]
  n_layers = states.shape[1]
  layers = list(range(n_layers)) if args.layers == "all" else \
      [int(x) for x in args.layers.split(",")]
  print(f"cache: {tuple(states.shape)}; rows by kind: {meta['kind'].value_counts().to_dict()}")

  # ------------------------------------------------------------------ readouts
  if "readouts" in stages:
    if mt is None and any(r != "probe" for r in args.readouts):
      mt = load_model(args)
    run_readouts(args, mt, states, meta, class_names, layers)

  # ------------------------------------------------------------------ eval
  if "eval" in stages:
    score_dir = os.path.join(args.out_dir, "scores")
    scores = {}
    for fname in sorted(os.listdir(score_dir)):
      if fname.endswith(".npz") and not fname.endswith("_entries.npz"):
        ls, s = load_scores(os.path.join(score_dir, fname))
        scores[fname[:-4]] = (ls, s)
    print(f"Evaluating {sorted(scores)} on all {len(class_names)} classes ...")
    summary = E.evaluate(scores, meta, class_names, args.n_boot, args.seed)
    summary.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    head = E.headline(summary, len(class_names))
    head.to_csv(os.path.join(args.out_dir, "headline.csv"), index=False)
    print(head.to_string(index=False))
    main_present = [r for r in MAIN_READOUTS if r in scores]
    E.plot(summary, args.out_dir, main_present)

    # Fair comparison: only classes every training-free readout can score and tell apart,
    # ranked among those classes only (the probe is evaluated on the same subset).
    keep, dropped = E.fair_classes(scores, [r for r in TRAINING_FREE if r in scores],
                                   class_names)
    with open(os.path.join(args.out_dir, "fair_classes.json"), "w", encoding="utf-8") as f:
      json.dump({"kept": [class_names[i] for i in keep], "dropped": dropped}, f, indent=2)
    print(f"\nFair subset: {len(keep)} of {len(class_names)} classes "
          f"(dropped: {', '.join(sorted(dropped)) or 'none'})")
    if len(keep) >= 2:
      sub_scores, sub_meta, sub_names = E.subset(scores, meta, class_names, keep)
      fair = E.evaluate(sub_scores, sub_meta, sub_names, args.n_boot, args.seed)
      fair.to_csv(os.path.join(args.out_dir, "summary_fair.csv"), index=False)
      head_fair = E.headline(fair, len(sub_names))
      head_fair.to_csv(os.path.join(args.out_dir, "headline_fair.csv"), index=False)
      print(f"{sub_meta['image_id'].nunique()} images, {len(sub_meta)} rows")
      print(head_fair.to_string(index=False))
      E.plot(fair, args.out_dir, main_present, prefix="fair_")
    print(f"All outputs in {args.out_dir}")


if __name__ == "__main__":
  main()
