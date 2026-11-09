r"""Figures, tables and number macros for the COCO-VTR report (no GPU, no model).

Everything is read from finished runs, so the report's numbers always match the result files:
  <vtr_dir>/samples.json, states.pt (meta only), scores/*.npz, summary_fair.csv, headline_fair.csv,
  fair_classes.json                         (run_vtr.py, the full LLaVA-1.5 benchmark)
  <causal_dir>/curve.csv, cases.csv, cases_by_layer.csv                    (causal_restore.py)

Outputs in --out_dir (copy the folder into the Overleaf project as generated/):
  numbers.tex        \newcommand macros used in the text (accuracies, hit rates, recoveries)
  tab_accuracy.tex   mean / layer-0 / best / half-peak per readout
  tab_examples.tex   real examples: image crop with the token marked, top-1 class per readout at
                     three layers (green = correct, red = wrong); ex_<i>.png are the crops
  tab_causal.tex     P(decodes | used) and P(decodes | not used) per readout
  fig_layers.pdf     top-1 by layer, all readouts, 95% image-clustered CI bands
  fig_causal.pdf     restore recovery by layer next to each readout's decode rate

Examples are drawn at random (--seed), not picked, so they are representative; pass --image_ids
to show specific images instead.

  python make_report_figures.py --coco_dir /projectnb/mlresearch/vishnuav/coco
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
from vtr import evaluate as E  # noqa: E402
from vtr.geometry import HIGH_COVERAGE  # noqa: E402

# (score / summary name, label, macro suffix, ours)
READOUTS = [
    ("logit_lens", "Logit lens", "LL", False),
    ("tuned_lens", "Tuned lens", "TL", False),
    ("embedding_lens", "Embedding lens", "EL", False),
    ("latentlens", "LatentLens", "LAT", False),
    ("patchscopes_pmi", "Patchscopes", "PS", False),
    ("selfie_pmi", "SelfIE", "SELFIE", False),
    ("image_slot_pmi", "Image-slot (ours)", "IS", True),
    ("probe_linear", "Linear probe (supervised)", "PROBE", False),
]
WORDS = {12: "Twelve", 16: "Sixteen", 18: "Eighteen", 20: "Twenty", 22: "TwentyTwo",
         24: "TwentyFour", 26: "TwentySix"}


def parse_args(argv=None):
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--vtr_dir", default="./results/vtr_llava15")
  p.add_argument("--causal_dir", default=None, help="default <vtr_dir>/causal_restore")
  p.add_argument("--coco_dir", required=True)
  p.add_argument("--out_dir", default="./results/report")
  p.add_argument("--n_examples", type=int, default=4)
  p.add_argument("--image_ids", default="", help="comma-separated COCO image ids to show")
  p.add_argument("--example_layers", default="2,16,28")
  p.add_argument("--seed", type=int, default=0)
  return p.parse_args(argv)


def fmt(x, digits=1):
  return f"{100 * x:.{digits}f}"


def common_layers(top1):
  per = [set(g["layer"]) for _, g in top1.groupby("readout")]
  return sorted(set.intersection(*per)) if per else []


# ----------------------------------------------------------------------------- accuracy
def accuracy(args, macros):
  summary = pd.read_csv(os.path.join(args.vtr_dir, "summary_fair.csv"))
  top1 = summary[(summary.metric == "top1") & (summary.condition == "object_high")]
  names = [r[0] for r in READOUTS if r[0] in set(top1.readout)]
  top1 = top1[top1.readout.isin(names)]
  layers = common_layers(top1)
  head = pd.read_csv(os.path.join(args.vtr_dir, "headline_fair.csv")).set_index("readout")
  rows = []
  for name, label, key, ours in READOUTS:
    if name not in names:
      continue
    g = top1[top1.readout == name].set_index("layer")
    mean = g.loc[layers, "mean"].mean()
    l0 = g.loc[0, "mean"] if 0 in g.index else np.nan
    best = int(g["mean"].idxmax())
    hp = head.loc[name, "half_peak_layer"] if name in head.index else np.nan
    macros[f"acc{key}"] = fmt(mean)
    macros[f"lzero{key}"] = fmt(l0)
    macros[f"best{key}"] = fmt(g.loc[best, "mean"])
    macros[f"bestlayer{key}"] = str(best)
    macros[f"halfpeak{key}"] = "--" if pd.isna(hp) else str(int(hp))
    rows.append((label, ours, mean, l0, g.loc[best, "mean"], best, hp))
  macros["nLayersMean"] = str(len(layers))
  macros["layersMean"] = f"{layers[0]}--{layers[-1]}" if layers else "--"
  wins = 0
  free = [r[0] for r in READOUTS if r[0] in names and r[0] not in ("image_slot_pmi",
                                                                   "probe_linear")]
  if "image_slot_pmi" in names:
    for l in layers:
      ours = top1[(top1.readout == "image_slot_pmi") & (top1.layer == l)]["mean"].iloc[0]
      best_other = top1[top1.readout.isin(free) & (top1.layer == l)]["mean"].max()
      wins += ours > best_other
  macros["isWins"] = str(wins)

  lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
           r"\begin{tabular}{lcccc}", r"\toprule",
           r"Readout & Mean & Layer 0 & Best (layer) & Half-peak layer \\", r"\midrule"]
  for label, ours, mean, l0, best_acc, best, hp in rows:
    lab = rf"\textbf{{{label}}}" if ours else label
    if label.startswith("Linear probe"):
      lines.append(r"\midrule")
    hp_s = "--" if pd.isna(hp) else str(int(hp))
    lines.append(f"{lab} & {fmt(mean)} & {fmt(l0)} & {fmt(best_acc)} ({best}) & {hp_s} \\\\")
  lines += [r"\bottomrule", r"\end{tabular}",
            rf"\caption{{Top-1 accuracy (\%) on object tokens of COCO-VTR (LLaVA-1.5-7B, "
            rf"63 classes, chance 1.6\%). Mean over the {len(layers)} layers every readout was "
            r"run on. Half-peak layer: first layer reaching half of that readout's own best "
            r"accuracy, a measure of how early it can read the object.}",
            r"\label{tab:accuracy}", r"\end{table}"]
  write(args, "tab_accuracy.tex", lines)
  plot_layers(args, top1, names)


def plot_layers(args, top1, names):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  colours = dict(zip([r[0] for r in READOUTS],
                     ["#1b2a41", "#5b6472", "#8c99ab", "#2a78d6", "#7fa3c9", "#1baf7a",
                      "#d9822b", "#9a5212"]))
  fig, ax = plt.subplots(figsize=(6.6, 3.0))
  for name, label, _, ours in READOUTS:
    if name not in names:
      continue
    g = top1[top1.readout == name].sort_values("layer")
    ls = "--" if name == "probe_linear" else "-"
    ax.plot(g.layer, 100 * g["mean"], ls, color=colours[name], lw=2.6 if ours else 1.4,
            label=label, zorder=3 if ours else 2)
    ax.fill_between(g.layer, 100 * g.ci_lo, 100 * g.ci_hi, color=colours[name], alpha=0.12,
                    lw=0)
  ax.set_xlabel("Layer of the visual token's hidden state")
  ax.set_ylabel("Top-1 accuracy (%)")
  ax.set_ylim(0, None)
  for side in ("top", "right"):
    ax.spines[side].set_visible(False)
  ax.grid(axis="y", color="#e3e6ea", lw=0.6)
  ax.legend(fontsize=7, ncol=2, frameon=False, loc="upper left")
  fig.tight_layout()
  fig.savefig(os.path.join(args.out_dir, "fig_layers.pdf"))
  plt.close(fig)


# ----------------------------------------------------------------------------- examples
def crop336(image):
  """What LLaVA-1.5 sees: shortest edge resized to 336, centre crop 336x336."""
  w, h = image.size
  s = 336 / min(w, h)
  image = image.resize((max(336, round(w * s)), max(336, round(h * s))))
  w, h = image.size
  left, top = (w - 336) // 2, (h - 336) // 2
  return image.crop((left, top, left + 336, top + 336))


def examples(args, macros):
  from PIL import Image, ImageDraw

  from visual_object_identification import ensure_image
  with open(os.path.join(args.vtr_dir, "samples.json"), encoding="utf-8") as f:
    data = json.load(f)
  class_names = data["class_names"]
  by_id = {s["image_id"]: s for s in data["samples"]}
  with open(os.path.join(args.vtr_dir, "fair_classes.json"), encoding="utf-8") as f:
    kept = json.load(f)["kept"]
  keep = [class_names.index(c) for c in kept]
  meta = torch.load(os.path.join(args.vtr_dir, "states.pt"), weights_only=False)["meta"]

  ok = ((meta["kind"] == "object") & (meta["coverage"] >= HIGH_COVERAGE)
        & meta["class_name"].isin(kept)).to_numpy()
  rng = np.random.RandomState(args.seed)
  if args.image_ids:
    ids = [int(x) for x in args.image_ids.split(",")]
  else:
    pool = meta.loc[ok, "image_id"].unique()
    ids = list(rng.choice(pool, size=min(args.n_examples, len(pool)), replace=False))
  rows = []
  for i in ids:
    cand = np.flatnonzero(ok & (meta["image_id"] == i).to_numpy())
    rows.append(int(rng.choice(cand)))

  show = [int(x) for x in args.example_layers.split(",")]
  scores = {}
  for name, *_ in READOUTS:
    path = os.path.join(args.vtr_dir, "scores", f"{name}.npz")
    if os.path.exists(path):
      scores[name] = run_vtr.load_scores(path)

  cols = []
  for k, r in enumerate(rows):
    m = meta.iloc[r]
    sample = by_id[m["image_id"]]
    img = crop336(Image.open(ensure_image(args.coco_dir, sample["file_name"])).convert("RGB"))
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    cov = np.asarray(sample["coverage"])
    for t in np.flatnonzero(cov >= HIGH_COVERAGE):
      y, x = divmod(int(t), 24)
      d.rectangle([x * 14, y * 14, x * 14 + 13, y * 14 + 13], fill=(42, 120, 214, 70))
    y, x = divmod(int(m["token"]), 24)
    d.rectangle([x * 14 - 2, y * 14 - 2, x * 14 + 15, y * 14 + 15], outline=(217, 130, 43, 255),
                width=4)
    Image.alpha_composite(img.convert("RGBA"), over).convert("RGB").save(
        os.path.join(args.out_dir, f"ex_{k}.png"))
    truth = kept.index(m["class_name"])
    cells = {}
    for name, (layers, s) in scores.items():
      words = []
      for l in show:
        if l not in layers:
          words.append(r"\textcolor{gray}{--}")
          continue
        row = s[layers.index(l), r][keep]
        if not np.isfinite(row).any():
          words.append(r"\textcolor{gray}{--}")
          continue
        top = int(np.nanargmax(row))
        words.append((r"\okw{%s}" if top == truth else r"\badw{%s}") % kept[top])
      cells[name] = words
    cols.append((m["class_name"], int(m["image_id"]), int(m["token"]), float(m["coverage"]),
                 cells))

  n = len(cols)
  lay = "/".join(str(l) for l in show)
  lines = [r"\begin{table*}[t]", r"\centering", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
           r"\renewcommand{\arraystretch}{1.25}",
           rf"\begin{{tabular}}{{l{'c' * n}}}", r"\toprule",
           " & " + " & ".join(rf"\includegraphics[width={0.92 / (n + 0.6):.3f}\textwidth]"
                              rf"{{generated/ex_{k}.png}}" for k in range(n)) + r" \\",
           r"\textbf{Ground truth} & " + " & ".join(rf"\textbf{{{c[0]}}}" for c in cols)
           + r" \\",
           r"\scriptsize COCO id, token & " + " & ".join(
               rf"\scriptsize {c[1]}, \#{c[2]} ({100 * c[3]:.0f}\% covered)" for c in cols)
           + r" \\", r"\midrule",
           rf"\multicolumn{{{n + 1}}}{{l}}{{\emph{{Top-1 class at layers {lay}}}}} \\"]
  for name, label, _, ours in READOUTS:
    if name not in scores:
      continue
    lab = rf"\textbf{{{label}}}" if ours else label
    if name == "probe_linear":
      lines.append(r"\midrule")
    lines.append(lab + " & " + " & ".join(r"\,/\,".join(c[4][name]) for c in cols) + r" \\")
  lines += [r"\bottomrule", r"\end{tabular}",
            rf"\caption{{Real outputs on {n} randomly drawn COCO-VTR tokens (seed {args.seed}; "
            r"not selected). The orange square is the visual token being read; blue shading "
            r"marks the object's tokens. Each cell gives the readout's top-1 class at layers "
            rf"{lay}: \okw{{green}} = correct, \badw{{red}} = wrong. All readouts rank the same 63 "
            r"class names.}", r"\label{tab:examples}", r"\end{table*}"]
  write(args, "tab_examples.tex", lines)


# ----------------------------------------------------------------------------- causal
def causal(args, macros):
  cdir = args.causal_dir or os.path.join(args.vtr_dir, "causal_restore")
  curve_path = os.path.join(cdir, "curve.csv")
  if not os.path.exists(curve_path):
    print(f"  no {curve_path}: causal outputs skipped")
    return
  curve = pd.read_csv(curve_path)
  obj = curve[curve["mode"] == "object"].set_index("layer")["recovery_mean"]
  for l, w in WORDS.items():
    if l in obj.index:
      macros[f"rec{w}"] = f"{100 * obj.loc[l]:.0f}"
  macros["recEarly"] = f"{100 * obj.loc[obj.index <= 12].mean():.0f}"
  images = pd.read_csv(os.path.join(cdir, "images.csv"))
  macros["nCausalImages"] = str(len(images))
  cases = pd.read_csv(os.path.join(cdir, "cases.csv")).set_index("readout")
  by_layer = pd.read_csv(os.path.join(cdir, "cases_by_layer.csv"))
  macros["nCausalValid"] = str(by_layer.groupby("layer")["n"].first().max())
  lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{lcc}", r"\toprule",
           r"Readout & $P(\text{decodes}\mid\text{used})$ & "
           r"$P(\text{decodes}\mid\text{not used})$ \\", r"\midrule"]
  for name, label, key, ours in READOUTS:
    if name not in cases.index:
      continue
    hu, hn = cases.loc[name, "hit_used"], cases.loc[name, "hit_unused"]
    macros[f"hitUsed{key}"], macros[f"hitUnused{key}"] = f"{hu:.2f}", f"{hn:.2f}"
    lab = rf"\textbf{{{label}}}" if ours else label
    lines.append(f"{lab} & {hu:.2f} & {hn:.2f} \\\\")
  lines += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Does a readout read the object where the model uses it? Over (image, "
            r"layer) pairs: \emph{used} = restoring the object's tokens at that layer brings "
            r"back at least half of the answer; \emph{decodes} = the readout's top-1 is the "
            r"object on at least half of the object's tokens.}", r"\label{tab:causal}",
            r"\end{table}"]
  write(args, "tab_causal.tex", lines)

  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  fig, (a, b) = plt.subplots(2, 1, figsize=(6.6, 4.4), sharex=True)
  for mode, colour, label in (("object", "#1b2a41", "restore the object's tokens"),
                              ("all_image", "#8c99ab", "restore all 576 image tokens")):
    g = curve[curve["mode"] == mode].sort_values("layer")
    if len(g):
      a.plot(g.layer, 100 * g.recovery_mean, color=colour, lw=2, label=label)
  a.set_ylabel("Answer recovered (%)")
  a.legend(fontsize=7, frameon=False)
  decode = by_layer.assign(rate=by_layer["A"] + by_layer["B"])
  for name, label, _, ours in READOUTS:
    g = decode[decode.readout == name].sort_values("layer")
    if len(g):
      b.plot(g.layer, 100 * g.rate, lw=2.6 if ours else 1.3, label=label,
             color="#d9822b" if ours else None)
  b.set_ylabel("Readout decodes (%)")
  b.set_xlabel("Layer")
  b.legend(fontsize=6.5, ncol=2, frameon=False)
  for ax in (a, b):
    for side in ("top", "right"):
      ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e3e6ea", lw=0.6)
  fig.tight_layout()
  fig.savefig(os.path.join(args.out_dir, "fig_causal.pdf"))
  plt.close(fig)


def write(args, name, lines):
  with open(os.path.join(args.out_dir, name), "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
  print(f"  wrote {name}")


def main(argv=None):
  args = parse_args(argv)
  os.makedirs(args.out_dir, exist_ok=True)
  macros = {}
  accuracy(args, macros)
  examples(args, macros)
  causal(args, macros)
  lines = ["% Generated by make_report_figures.py from the result files; do not edit."]
  lines += [rf"\def\{k}{{{v}}}" for k, v in sorted(macros.items())]
  write(args, "numbers.tex", lines)
  print(f"All outputs in {args.out_dir}")


if __name__ == "__main__":
  main()
