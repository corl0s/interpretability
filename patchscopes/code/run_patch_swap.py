r"""Task 6 (part 2): patch-swap locality control on LLaVA-1.5.

Images with two different objects A and B. For tokens on A we ask: does a readout decode A (the
patch's own object) or B (the other object in the same image)? A readout that reads the patch
ranks A above B; one that reads image-level information has no reason to prefer A.

Selection (COCO val2017): two non-crowd objects of different classes from the fair class set,
each the only instance of its class, each with >= --min_tokens tokens at coverage >= 0.5; tokens
touched by both objects are excluded. Rows per object: --per_object high-coverage tokens and the
mean-pooled state, at all layers.

Readouts (same scoring as run_vtr.py, ranked over the fair classes): logit lens, LatentLens,
Patchscopes (identity, PMI), SelfIE (PMI), and the linear probe trained on the main COCO-VTR cache
(it never saw these images).

Metrics per readout, layer and row kind (token / pooled), with image-clustered CIs:
  own_over_other  own class ranked above the other object's class (chance 0.5)
  top1_own        top-1 is the own class
  top1_other      top-1 is the other object's class (the "follows the image" error)

  python run_patch_swap.py --coco_dir /projectnb/mlresearch/vishnuav/coco
"""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_object_identification import bootstrap_by_image  # noqa: E402
from run_failure_map import SHORT  # noqa: E402
from visual_object_identification import (ann_to_mask, capture_visual_states,  # noqa: E402
                                          ensure_annotations, ensure_image)
from vtr import evaluate as E  # noqa: E402
from vtr.geometry import HIGH_COVERAGE, GEOMETRIES  # noqa: E402
from vtr.readouts import LatentLens, LogitLens, Patchscopes, class_forms  # noqa: E402

READOUTS = ["logit_lens", "latentlens", "patchscopes_pmi", "selfie_pmi", "probe_linear"]


def select_pairs(ann_path, geometry, classes, n_images, min_tokens, min_area, seed):
  with open(ann_path, encoding="utf-8") as f:
    coco = json.load(f)
  cats = {c["id"]: c["name"] for c in coco["categories"]}
  imgs = {im["id"]: im for im in coco["images"]}
  by_image = defaultdict(list)
  for a in coco["annotations"]:
    by_image[a["image_id"]].append(a)
  allowed = set(classes)
  candidates = []
  for image_id in sorted(by_image):
    anns = by_image[image_id]
    counts = Counter(a["category_id"] for a in anns)
    pool = [a for a in anns if not a["iscrowd"] and counts[a["category_id"]] == 1
            and cats[a["category_id"]] in allowed and a["area"] >= min_area
            and not isinstance(a["segmentation"], dict)]
    if len(pool) < 2:
      continue
    info = imgs[image_id]
    covs = []
    for a in sorted(pool, key=lambda a: -a["area"])[:4]:
      m = ann_to_mask(a, info["height"], info["width"])
      if m is not None and m.any():
        covs.append((a, geometry.coverage(m)))
    best = None
    for i in range(len(covs)):
      for j in range(i + 1, len(covs)):
        (a, ca), (b, cb) = covs[i], covs[j]
        shared = (ca > 0) & (cb > 0)
        na = int(((ca >= HIGH_COVERAGE) & ~shared).sum())
        nb = int(((cb >= HIGH_COVERAGE) & ~shared).sum())
        if na >= min_tokens and nb >= min_tokens and (best is None or min(na, nb) > best[0]):
          best = (min(na, nb), a, ca, b, cb, shared)
    if best is None:
      continue
    _, a, ca, b, cb, shared = best
    candidates.append({
        "image_id": image_id, "file_name": info["file_name"],
        "objects": [
            {"class_name": cats[a["category_id"]],
             "tokens": np.flatnonzero((ca >= HIGH_COVERAGE) & ~shared).tolist()},
            {"class_name": cats[b["category_id"]],
             "tokens": np.flatnonzero((cb >= HIGH_COVERAGE) & ~shared).tolist()}]})
  random.Random(seed).shuffle(candidates)
  return candidates[:n_images]


def cache(mt, processor, pairs, coco_dir, per_object, seed):
  rng = np.random.RandomState(seed)
  states, meta = [], []
  for idx, p in enumerate(pairs):
    image = Image.open(ensure_image(coco_dir, p["file_name"])).convert("RGB")
    full = capture_visual_states(mt, processor, image, verify=(idx == 0))   # [L, T, d]
    for slot, (own, other) in enumerate([(0, 1), (1, 0)]):
      o, x = p["objects"][own], p["objects"][other]
      toks = rng.choice(o["tokens"], size=min(per_object, len(o["tokens"])), replace=False)
      base = {"image_id": p["image_id"], "slot": slot, "class_name": o["class_name"],
              "other_class": x["class_name"]}
      for t in toks:
        states.append(full[:, int(t)])
        meta.append({**base, "kind": "token", "token": int(t)})
      states.append(full[:, torch.as_tensor(o["tokens"])].float().mean(1).half())
      meta.append({**base, "kind": "pooled_mean", "token": -1})
    if (idx + 1) % 25 == 0 or idx == 0:
      print(f"  cached {idx + 1}/{len(pairs)} images ({len(meta)} rows)", flush=True)
  return torch.stack(states), pd.DataFrame(meta)


def train_probe(vtr_dir, class_names, layers, seed):
  """Linear probes per layer on the main COCO-VTR object tokens (coverage >= 0.5)."""
  from sklearn.linear_model import LogisticRegression
  from sklearn.pipeline import make_pipeline
  from sklearn.preprocessing import StandardScaler
  c = torch.load(os.path.join(vtr_dir, "states.pt"), weights_only=False)
  st, m = c["states"], c["meta"]
  keep = ((m["kind"] == "object") & (m["coverage"] >= HIGH_COVERAGE)
          & m["class_name"].isin(class_names)).to_numpy()
  y = m.loc[keep, "class_name"].map({n: i for i, n in enumerate(class_names)}).to_numpy()
  probes = {}
  for l in layers:
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    clf.fit(st[keep, l].float().numpy(), y)
    probes[l] = clf
    print(f"    probe layer {l} trained on {keep.sum()} main-cache tokens", flush=True)
  return probes


def score_all(args, mt, states, meta, class_names, layers):
  out = {}
  forms = class_forms(class_names, synonyms=False)
  for name in args.readouts:
    path = os.path.join(args.out_dir, "scores", f"{name}.npy")
    if os.path.exists(path):
      out[name] = np.load(path)
      continue
    t0 = time.time()
    if name == "logit_lens":
      ro = LogitLens(mt, class_names, forms, args.device)
      s = np.stack([ro.scores(states[:, l].float()) for l in layers])
    elif name == "latentlens":
      ro = LatentLens(args.latentlens_bank, mt.tokenizer, class_names, forms, args.device,
                      cache_file=os.path.join(args.out_dir, "scores", "latentlens_entries.npz"))
      s = np.stack([ro.scores(states[:, l].float()) for l in layers])
    elif name in ("patchscopes_pmi", "selfie_pmi"):
      prompt = "identity" if name == "patchscopes_pmi" else "selfie"
      ro = Patchscopes(mt, class_names, forms, prompt, args.device,
                       rows_per_batch=args.rows_per_batch)
      s = np.stack([ro.scores(states[:, l].float(), l)[1] for l in layers])
    elif name == "probe_linear":
      probes = train_probe(args.vtr_dir, class_names, layers, args.seed)
      s = []
      for l in layers:
        proba = probes[l].predict_proba(states[:, l].float().numpy())
        block = np.full((len(meta), len(class_names)), -np.inf, dtype=np.float32)
        block[:, probes[l].classes_] = np.log(proba + 1e-12)
        s.append(block)
      s = np.stack(s)
    else:
      raise ValueError(name)
    np.save(path, s.astype(np.float32))
    out[name] = s
    print(f"  {name}: {time.time() - t0:.0f}s", flush=True)
  return out


def evaluate(scores, meta, class_names, layers, n_boot, seed):
  idx = {c: i for i, c in enumerate(class_names)}
  own = meta["class_name"].map(idx).to_numpy()
  other = meta["other_class"].map(idx).to_numpy()
  frames = []
  for name, s in scores.items():
    for kind in ("token", "pooled_mean"):
      mask = (meta["kind"] == kind).to_numpy()
      cols = {"own_over_other": [], "top1_own": [], "top1_other": []}
      for i, l in enumerate(layers):
        sl = s[i][mask]
        r = np.arange(mask.sum())
        cols["own_over_other"].append(sl[r, own[mask]] > sl[r, other[mask]])
        cols["top1_own"].append(E.ranks(sl, own[mask]) == 1)
        cols["top1_other"].append(E.ranks(sl, other[mask]) == 1)
      for metric, vals in cols.items():
        long = pd.DataFrame({"image_id": np.tile(meta["image_id"].to_numpy()[mask], len(layers)),
                             "layer": np.repeat(layers, mask.sum()),
                             "score": np.concatenate(vals).astype(float)})
        b = bootstrap_by_image(long, "score", n_boot, seed)
        b["readout"], b["kind"], b["metric"] = name, kind, metric
        frames.append(b)
  return pd.concat(frames, ignore_index=True)


def plot(summary, path):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"]
  fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
  for ax, kind in zip(axes, ("token", "pooled_mean")):
    s = summary[(summary.kind == kind) & (summary.metric == "own_over_other")]
    for i, (name, g) in enumerate(s.groupby("readout", sort=False)):
      ax.plot(g.layer, g["mean"], color=palette[i % len(palette)], lw=2, label=SHORT[name])
      ax.fill_between(g.layer, g.ci_lo, g.ci_hi, color=palette[i % len(palette)], alpha=0.12,
                      lw=0)
    ax.axhline(0.5, color="#9a9994", ls=":", lw=1)
    ax.set_title("single tokens" if kind == "token" else "mean-pooled object", loc="left")
    ax.set_xlabel("LLM layer")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", alpha=0.3)
    for side in ("top", "right"):
      ax.spines[side].set_visible(False)
  axes[0].set_ylabel("own object ranked above the other object")
  axes[0].legend(frameon=False, fontsize=8)
  fig.suptitle("Patch swap: does the readout follow the patch or the image? (chance 50%)",
               x=0.01, ha="left")
  fig.tight_layout()
  fig.savefig(path, dpi=150)
  plt.close(fig)


def parse_args(argv=None):
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--coco_dir", default=None)
  p.add_argument("--vtr_dir", default="./results/vtr_llava15",
                 help="main COCO-VTR run (fair_classes.json, states.pt for the probe)")
  p.add_argument("--out_dir", default="./results/vtr_llava15/patch_swap")
  p.add_argument("--model_name", default="llava-hf/llava-1.5-7b-hf")
  p.add_argument("--latentlens_bank", default="./results/latentlens_index/bank")
  p.add_argument("--n_images", type=int, default=300)
  p.add_argument("--min_tokens", type=int, default=4)
  p.add_argument("--min_area", type=float, default=4000)
  p.add_argument("--per_object", type=int, default=4)
  p.add_argument("--layers", default="0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30")
  p.add_argument("--readouts", nargs="+", default=READOUTS, choices=READOUTS)
  p.add_argument("--rows_per_batch", type=int, default=512)
  p.add_argument("--n_boot", type=int, default=1000)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  return p.parse_args(argv)


def main(argv=None, mt=None):
  args = parse_args(argv)
  os.makedirs(os.path.join(args.out_dir, "scores"), exist_ok=True)
  layers = [int(x) for x in args.layers.split(",")]
  with open(os.path.join(args.vtr_dir, "fair_classes.json"), encoding="utf-8") as f:
    class_names = json.load(f)["kept"]

  pairs_path = os.path.join(args.out_dir, "pairs.json")
  if not os.path.exists(pairs_path):
    pairs = select_pairs(ensure_annotations(args.coco_dir), GEOMETRIES["llava-1.5"](),
                         class_names, args.n_images, args.min_tokens, args.min_area, args.seed)
    with open(pairs_path, "w", encoding="utf-8") as f:
      json.dump(pairs, f)
  with open(pairs_path, encoding="utf-8") as f:
    pairs = json.load(f)
  print(f"{len(pairs)} two-object images, {len(class_names)} candidate classes")

  if mt is None:
    from general_utils import ModelAndTokenizer
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    mt = ModelAndTokenizer(args.model_name, torch_dtype=dtype, device=args.device)
  cache_path = os.path.join(args.out_dir, "states.pt")
  if not os.path.exists(cache_path):
    states, meta = cache(mt, mt.processor, pairs, args.coco_dir, args.per_object, args.seed)
    torch.save({"states": states, "meta": meta}, cache_path)
  c = torch.load(cache_path, weights_only=False)
  states, meta = c["states"], c["meta"]
  print(f"cache {tuple(states.shape)}; {meta['kind'].value_counts().to_dict()}")

  scores = score_all(args, mt, states, meta, class_names, layers)
  summary = evaluate(scores, meta, class_names, layers, args.n_boot, args.seed)
  summary.to_csv(os.path.join(args.out_dir, "patch_swap_summary.csv"), index=False)
  plot(summary, os.path.join(args.out_dir, "patch_swap.png"))
  view = summary[(summary.kind == "token")].pivot_table(
      index=["readout", "metric"], columns="layer", values="mean").round(3)
  pd.set_option("display.width", 250)
  print("\nSingle tokens (own_over_other: chance 0.5):")
  print(view.to_string())
  print(f"\nAll outputs in {args.out_dir}")


if __name__ == "__main__":
  main()
