r"""Where do readouts get their answer from? Diagnostics on the saved COCO-VTR scores (CPU only).

Every table is closed-set top-1 over the fair classes, per readout, averaged over three layer
windows (fractions of depth, matching the causal check on LLaVA: the model uses the object
tokens up to ~layer 17 and not from ~20 on): early (<= 0.4 L), mid, late (>= 0.625 L).

  conditions  object tokens (coverage >= 0.5), outside tokens (on background / on another
              object), the mean-pooled object state, the random control. An outside token is
              scored against the TARGET object's class: a readout that names it there reads
              the image, not the token. "outside_local" = it names the object actually under
              that outside token instead.
  hardness    object tokens split by whether the target is the image's dominant object (the
              COCO class with the largest total area; "easy") or not ("hard"), and the same with
              LLaVA's own main-object answer where that file exists (--main_object)
  errors      what the wrong answers on object tokens are, in this order: the image's dominant
              object, another object in the image, the target's COCO supercategory, other. Next
              to each share, the share a uniformly random wrong answer would get ("chance_*").
  agreement   per object token: right for no readout / only the probe / the probe and another /
              all; and P(a readout is right | the probe is wrong) -- information present but
              hard to read vs absent
  edge        object tokens by distance (in tokens) to the nearest token the object does not
              touch: 1 = on the boundary
  size        object tokens by COCO size bucket of the target (small / medium / large)
  bins        object tokens by overlap bin (share of the token's region the object covers)

  python diagnose_readouts.py --vtr_dir ./results/vtr_llava15 \
      --coco_dir /projectnb/mlresearch/vishnuav/coco
    -> <vtr_dir>/diagnose/{refs.csv, long.csv, <table>.csv}
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_vtr  # noqa: E402
from train_oracle import load_benchmark  # noqa: E402
from visual_object_identification import ann_to_mask, ensure_annotations  # noqa: E402
from vtr import evaluate as E  # noqa: E402
from vtr.data import load_coco  # noqa: E402
from vtr.geometry import BINS, GEOMETRIES, HIGH_COVERAGE  # noqa: E402

READOUTS = [r for r in run_vtr.MAIN_READOUTS if r != "tuned_lens"]
PROBE = "probe_linear"
WINDOWS = ("early", "mid", "late")


def parse_args(argv=None):
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--vtr_dir", default="./results/vtr_llava15")
  p.add_argument("--coco_dir", required=True, help="for the annotations of the other objects")
  p.add_argument("--out_dir", default=None, help="default <vtr_dir>/diagnose")
  p.add_argument("--readouts", default=",".join(READOUTS))
  p.add_argument("--main_object", default=None,
                 help="LLaVA's main-object answers (image_id, main_object); default "
                      "<vtr_dir>/oracle_crop7_add/eval/main_object.csv if it exists")
  p.add_argument("--min_pixels", type=int, default=56 * 56, help="Qwen2.5-VL geometry")
  p.add_argument("--max_pixels", type=int, default=12845056, help="Qwen2.5-VL geometry")
  return p.parse_args(argv)


def window(layer, n_layers):
  d = layer / max(n_layers - 1, 1)
  return "early" if d <= 0.4 else "late" if d >= 0.625 else "mid"


# ---------------------------------------------------------------------------- references


def image_refs(args, samples, geometry):
  """Per image: dominant class, classes present, and per token the class of the other object
  under it (largest coverage >= 0.5 among the non-target annotations, else '')."""
  coco, cats, _ = load_coco(ensure_annotations(args.coco_dir))
  imgs = {im["id"]: im for im in coco["images"]}
  by_image = defaultdict(list)
  for a in coco["annotations"]:
    by_image[a["image_id"]].append(a)
  refs = {}
  for s in samples:
    anns, info = by_image[s["image_id"]], imgs[s["image_id"]]
    area = defaultdict(float)
    for a in anns:
      area[cats[a["category_id"]]["name"]] += a["area"]
    n_tok = len(s["coverage"])
    best, local = np.zeros(n_tok), np.full(n_tok, "", dtype=object)
    for a in anns:
      if a["id"] == s["ann_id"]:
        continue
      mask = ann_to_mask(a, info["height"], info["width"])  # None for crowd (RLE)
      if mask is None or not mask.any():
        continue
      cov = geometry.coverage(mask)
      take = (cov >= HIGH_COVERAGE) & (cov > best)
      best[take], local[take] = cov[take], cats[a["category_id"]]["name"]
    refs[s["image_id"]] = {"dominant": max(area, key=area.get), "present": set(area),
                           "local": local}
  return refs, {c["name"]: c["supercategory"] for c in cats.values()}


def edge_distance(sample):
  """Per token: distance (in tokens) to the nearest token the object does not touch."""
  from scipy.ndimage import distance_transform_edt
  rows, cols = sample["grid"]
  inside = np.asarray(sample["coverage"]).reshape(rows, cols) > 0
  return distance_transform_edt(inside).reshape(-1)


def row_table(args, meta, samples, kept, geometry):
  """One row per cached row of a fair-class image, with every grouping the tables need."""
  out_csv = os.path.join(args.out_dir, "refs.csv")
  if os.path.exists(out_csv):
    return text_columns(pd.read_csv(out_csv, dtype={c: str for c in TEXT}))
  by_id = {s["image_id"]: s for s in samples}
  refs, supercat = image_refs(args, [s for s in samples if s["class_name"] in kept], geometry)
  main = {}
  path = args.main_object or os.path.join(args.vtr_dir, "oracle_crop7_add", "eval",
                                          "main_object.csv")
  if os.path.exists(path):
    df = pd.read_csv(path, keep_default_na=False)
    main = dict(zip(df["image_id"], df["main_object"]))
    print(f"  LLaVA main-object answers for {len(main)} images ({path})")
  rows = []
  dist = {}
  for i, m in meta.iterrows():
    if m["class_name"] not in kept or m["image_id"] not in refs:
      continue
    ref, s = refs[m["image_id"]], by_id[m["image_id"]]
    tok = int(m["token"])
    if m["image_id"] not in dist:
      dist[m["image_id"]] = edge_distance(s)
    d = dist[m["image_id"]][tok] if tok >= 0 else np.nan
    rows.append({"row": i, "image_id": m["image_id"], "kind": m["kind"], "bin": m["bin"],
                 "size": m["size"], "coverage": m["coverage"], "class_name": m["class_name"],
                 "supercategory": supercat[m["class_name"]], "dominant": ref["dominant"],
                 "present": "|".join(sorted(ref["present"])),
                 "local": ref["local"][tok] if tok >= 0 else "",
                 "llava_main": main.get(m["image_id"], ""),
                 "edge": "" if np.isnan(d) else ("4+" if d >= 4 else str(int(d)))})
  df = pd.DataFrame(rows)
  df.to_csv(out_csv, index=False)
  return text_columns(df)


TEXT = ("local", "llava_main", "edge", "present", "dominant", "bin", "size", "class_name",
        "supercategory", "kind")


def text_columns(df):
  for c in TEXT:
    df[c] = df[c].fillna("").astype(str)
  return df


# ---------------------------------------------------------------------------------- tables


def readout_hits(args, refs, kept, class_names):
  """{name: (layers, correct [L, n] bool, pred [L, n] index into kept, scored [n] bool)}."""
  index = {c: i for i, c in enumerate(class_names)}
  keep = [index[c] for c in kept]
  lab = refs["class_name"].map({c: i for i, c in enumerate(kept)}).to_numpy()
  rows = refs["row"].to_numpy()
  out = {}
  for name in args.readouts.split(","):
    path = os.path.join(args.vtr_dir, "scores", f"{name}.npz")
    if not os.path.exists(path):
      print(f"  (no {path}: {name} skipped)")
      continue
    layers, scores = run_vtr.load_scores(path)
    s = scores[:, rows][:, :, keep]
    scored = ~np.isnan(s[0]).all(axis=1)
    s = np.nan_to_num(s, nan=-np.inf)
    out[name] = (layers, np.stack([E.ranks(x, lab) == 1 for x in s]), s.argmax(-1), scored)
  return out


def rates(hits, masks, n_layers, value="correct"):
  """Long table: readout, layer, window, group, rate, n, from boolean row masks."""
  rows = []
  for name, (layers, correct, pred, scored) in hits.items():
    v = value(correct, pred) if callable(value) else correct
    for g, mask in masks.items():
      mask = mask & scored
      if not mask.any():
        continue
      for li, l in enumerate(layers):
        rows.append({"readout": name, "layer": int(l), "window": window(l, n_layers),
                     "group": g, "rate": float(v[li][mask].mean()), "n": int(mask.sum())})
  return pd.DataFrame(rows)


def by_window(long, groups=None):
  """readout x (group, window) means of the per-layer rates."""
  t = long.groupby(["readout", "group", "window"])["rate"].mean().unstack(["group", "window"])
  groups = groups or list(dict.fromkeys(long["group"]))
  cols = [(g, w) for g in groups for w in WINDOWS if (g, w) in t.columns]
  return t[cols]


def error_table(hits, refs, kept, n_layers, base):
  """Shares of wrong answers by type (and chance), per readout and window, on `base` rows."""
  K = len(kept)
  ki = {c: i for i, c in enumerate(kept)}
  lab = refs["class_name"].map(ki).to_numpy()
  dom = refs["dominant"].map(lambda c: ki.get(c, -1)).to_numpy()
  dom[dom == lab] = -1
  present = [{ki[c] for c in p.split("|") if c in ki} for p in refs["present"]]
  sup = {c: s for c, s in zip(refs["class_name"], refs["supercategory"])}
  sup_of = np.array([sup.get(c, "") for c in kept])
  targ_sup = refs["supercategory"].to_numpy()
  # chance: a uniformly random wrong class, typed in the same order
  ch = np.zeros((len(refs), 3))
  for r in range(len(refs)):
    others = present[r] - {lab[r], dom[r]}
    same = {k for k in range(K) if sup_of[k] == targ_sup[r]} - {lab[r], dom[r]} - others
    ch[r] = [dom[r] >= 0, len(others), len(same)]
  ch /= max(K - 1, 1)
  rows = []
  for name, (layers, correct, pred, scored) in hits.items():
    for li, l in enumerate(layers):
      wrong = base & scored & ~correct[li]
      if not wrong.any():
        continue
      p = pred[li]
      is_dom = (p == dom) & (dom >= 0)
      in_img = np.array([p[r] in present[r] for r in range(len(p))]) & ~is_dom
      same_sup = (sup_of[p] == targ_sup) & ~is_dom & ~in_img
      kinds = {"dominant": is_dom, "in_image": in_img, "supercategory": same_sup}
      row = {"readout": name, "layer": int(l), "window": window(l, n_layers),
             "n_wrong": int(wrong.sum())}
      for k, (key, m) in enumerate(kinds.items()):
        row[key] = float(m[wrong].mean())
        row[f"chance_{key}"] = float(ch[wrong, k].mean())
      row["other"] = 1.0 - row["dominant"] - row["in_image"] - row["supercategory"]
      rows.append(row)
  return pd.DataFrame(rows)


def agreement_table(hits, n_layers, base):
  """Per layer, over object tokens all readouts scored: who is right."""
  if PROBE not in hits:
    return pd.DataFrame()
  names = [n for n in hits if n != PROBE]
  rows = []
  layers = hits[PROBE][0]
  for li, l in enumerate(layers):
    have = [n for n in names if l in hits[n][0]]
    mask = base & hits[PROBE][3]
    for n in have:
      mask = mask & hits[n][3]
    if not mask.any() or not have:
      continue
    probe = hits[PROBE][1][li][mask]
    other = np.stack([hits[n][1][hits[n][0].index(l)][mask] for n in have])
    row = {"layer": int(l), "window": window(l, n_layers), "n": int(mask.sum()),
           "readouts": len(have) + 1,
           "none": float((~probe & ~other.any(0)).mean()),
           "probe_only": float((probe & ~other.any(0)).mean()),
           "probe_and_other": float((probe & other.any(0) & ~other.all(0)).mean()),
           "all": float((probe & other.all(0)).mean()),
           "other_not_probe": float((~probe & other.any(0)).mean())}
    for n, o in zip(have, other):
      row[f"right_if_probe_wrong:{n}"] = float(o[~probe].mean()) if (~probe).any() else np.nan
    rows.append(row)
  return pd.DataFrame(rows)


def main(argv=None):
  args = parse_args(argv)
  args.out_dir = args.out_dir or os.path.join(args.vtr_dir, "diagnose")
  os.makedirs(args.out_dir, exist_ok=True)
  data, kept = load_benchmark(args.vtr_dir)
  name = data.get("geometry", "llava-1.5")
  geometry = (GEOMETRIES[name](args.min_pixels, args.max_pixels) if name == "qwen2.5-vl"
              else GEOMETRIES[name]())
  cache = torch.load(os.path.join(args.vtr_dir, "states.pt"), weights_only=False)
  n_layers = cache["states"].shape[1]
  refs = row_table(args, cache["meta"], data["samples"], kept, geometry)
  del cache
  hits = readout_hits(args, refs, kept, data["class_names"])
  if not hits:
    raise SystemExit("no readout scores found")
  pd.set_option("display.width", 250)
  pd.set_option("display.max_columns", 60)

  kind = refs["kind"].to_numpy()
  obj = (kind == "object") & (refs["coverage"].astype(float) >= HIGH_COVERAGE).to_numpy()
  outside = kind == "outside"
  on_other = refs["local"].astype(str).to_numpy() != ""
  easy = (refs["class_name"] == refs["dominant"]).to_numpy()
  llava = refs["llava_main"].astype(str).to_numpy()
  ki = {c: i for i, c in enumerate(kept)}
  local_idx = refs["local"].map(lambda c: ki.get(c, -1)).to_numpy()

  tables = {}
  tables["conditions"] = by_window(pd.concat([
      rates(hits, {"object": obj, "outside": outside,
                   "outside_background": outside & ~on_other,
                   "outside_on_object": outside & on_other,
                   "pooled_mean": kind == "pooled_mean", "random": kind == "random"}, n_layers),
      rates(hits, {"outside_local": outside & (local_idx >= 0)}, n_layers,
            value=lambda c, p: p == local_idx[None]),
  ]))
  hard_masks = {"easy": obj & easy, "hard": obj & ~easy}
  if (llava != "").any():
    named = llava != ""
    hard_masks.update({"easy_llava": obj & named & (refs["class_name"].to_numpy() == llava),
                       "hard_llava": obj & named & (refs["class_name"].to_numpy() != llava)})
  tables["hardness"] = by_window(rates(hits, hard_masks, n_layers))
  tables["edge"] = by_window(rates(hits, {f"edge_{e}": obj & (refs["edge"].astype(str) == e)
                                          for e in ("1", "2", "3", "4+")}, n_layers))
  tables["size"] = by_window(rates(hits, {s: obj & (refs["size"] == s).to_numpy()
                                          for s in ("small", "medium", "large")}, n_layers))
  tables["bins"] = by_window(rates(hits, {f"bin_{b}": (kind == "object")
                                          & (refs["bin"] == b).to_numpy()
                                          for b, _, _ in BINS}, n_layers))
  errors = {"all": error_table(hits, refs, kept, n_layers, obj),
            "hard": error_table(hits, refs, kept, n_layers, obj & ~easy)}
  agree = agreement_table(hits, n_layers, obj)

  for key, t in tables.items():
    t.to_csv(os.path.join(args.out_dir, f"{key}.csv"))
  for key, t in errors.items():
    t.to_csv(os.path.join(args.out_dir, f"errors_{key}.csv"), index=False)
  agree.to_csv(os.path.join(args.out_dir, "agreement.csv"), index=False)

  n = {"object tokens": int(obj.sum()), "hard object tokens": int((obj & ~easy).sum()),
       "outside tokens": int(outside.sum()), "outside on another object": int((outside & on_other).sum())}
  print(f"\n{len(kept)} fair classes; " + ", ".join(f"{k} {v}" for k, v in n.items()))
  print("windows: early <= 0.4 depth, late >= 0.625 depth (LLaVA: 0-12 / 13-19 / 20-31)")
  for key, t in tables.items():
    print(f"\n== {key}: top-1 (%), mean over the window's layers")
    print((t * 100).round(1).to_string())
  for key, t in errors.items():
    if t.empty:
      continue
    cols = ["dominant", "chance_dominant", "in_image", "chance_in_image", "supercategory",
            "chance_supercategory", "other"]
    print(f"\n== wrong answers on {key} object tokens: share by type (%), vs a random wrong class")
    print((t.groupby(["readout", "window"])[cols].mean() * 100).round(1).unstack("window")
          .reindex(columns=WINDOWS, level=1).to_string())
  if not agree.empty:
    cols = ["none", "probe_only", "probe_and_other", "all", "other_not_probe"]
    print("\n== agreement on object tokens (%): who is right")
    print((agree.groupby("window")[cols].mean() * 100).round(1).reindex(WINDOWS).to_string())
    cond = [c for c in agree.columns if c.startswith("right_if_probe_wrong:")]
    print("\n== P(readout right | probe wrong) (%)")
    t = (agree.groupby("window")[cond].mean() * 100).round(1).reindex(WINDOWS).T
    t.index = [c.split(":", 1)[1] for c in t.index]
    print(t.to_string())
  with open(os.path.join(args.out_dir, "counts.json"), "w", encoding="utf-8") as f:
    json.dump(n, f, indent=1)
  print(f"\nAll outputs in {args.out_dir}")


if __name__ == "__main__":
  main()
