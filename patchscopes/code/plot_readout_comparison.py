"""Readout comparison figure for the image-slot injection result (LLaVA-1.5-7B, fair subset).

Left: image-slot injection vs the other training-free readouts and the probe, by source layer.
Right: image-slot variants vs the SelfIE text-slot baseline.
Injection numbers: run_injection_proto.py --summarize_only (test images, dev-chosen target).
Other readouts: summary_fair.csv (object_high, top-1, all benchmark rows).

  python plot_readout_comparison.py --vtr_dir ./results/vtr_llava15 --out readout_comparison.png
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LAYERS = [0, 4, 8, 12, 16, 20, 24, 28]
# From run_injection_proto.py --summarize_only (2026-10-16), used when proto_best.csv is absent.
PROTO = {
    "selfie_text": [0.319, 0.315, 0.330, 0.296, 0.300, 0.313, 0.351, 0.315],
    "selfie_text_norm": [0.275, 0.334, 0.338, 0.283, 0.304, 0.334, 0.357, 0.298],
    "img_all": [0.461, 0.457, 0.471, 0.410, 0.423, 0.486, 0.490, 0.480],
    "img_all_norm": [0.459, 0.465, 0.457, 0.402, 0.406, 0.459, 0.427, 0.334],
    "img_all_multi": [0.374, 0.431, 0.469, 0.433, 0.467, 0.495, 0.497, 0.495],
}
INK, MUTED, GRID = "#17191c", "#6b6b66", "#e4e3de"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"


def load(vtr_dir):
  best = os.path.join(vtr_dir, "injection_proto", "proto_best.csv")
  if os.path.exists(best):
    b = pd.read_csv(best)
    proto = {v: g.set_index("source")["test_object"].reindex(LAYERS).tolist()
             for v, g in b.groupby("variant")}
  else:
    proto = PROTO
  s = pd.read_csv(os.path.join(vtr_dir, "summary_fair.csv"))
  s = s[(s.condition == "object_high") & (s.metric == "top1")]
  tools = s.pivot_table(index="layer", columns="readout", values="mean").reindex(LAYERS)
  return proto, tools


def style(ax):
  for side in ("top", "right"):
    ax.spines[side].set_visible(False)
  for side in ("left", "bottom"):
    ax.spines[side].set_color(MUTED)
  ax.tick_params(colors=MUTED, labelsize=9)
  ax.grid(axis="y", color=GRID, lw=0.8)
  ax.set_axisbelow(True)
  ax.set_xticks(LAYERS)
  ax.set_ylim(0, 0.72)
  ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
  ax.set_xlabel("source layer", color=MUTED, fontsize=9)


def line(ax, y, color, label, marker, lw=2, ls="-", end_dy=0.0):
  ax.plot(LAYERS, y, ls, color=color, lw=lw, marker=marker, ms=6, mec="white", mew=1,
          label=label, zorder=3)
  ax._end_labels = getattr(ax, "_end_labels", []) + [(float(y[-1]), label, color, marker)]


def place_labels(ax, gap=0.034):
  """End-of-line labels, nudged apart vertically so none overlap; marker keys the series."""
  items = sorted(ax._end_labels)
  ys = [v for v, *_ in items]
  for _ in range(50):
    for i in range(1, len(ys)):
      if ys[i] - ys[i - 1] < gap:
        mid = (ys[i] + ys[i - 1]) / 2
        ys[i - 1], ys[i] = mid - gap / 2, mid + gap / 2
  for (v, label, color, marker), y in zip(items, ys):
    ax.plot([29.6], [y], marker=marker, color=color, ms=6, mec="white", mew=1, clip_on=False)
    ax.annotate(label, (29.6, y), xytext=(7, 0), textcoords="offset points", va="center",
                fontsize=8.5, color=INK, annotation_clip=False)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--vtr_dir", default="./results/vtr_llava15")
  p.add_argument("--out", default="readout_comparison.png")
  p.add_argument("--left_only", action="store_true", help="only the readout comparison panel")
  args = p.parse_args()
  proto, tools = load(args.vtr_dir)
  chance = 1 / 63

  if args.left_only:
    fig, a = plt.subplots(figsize=(9, 5))
    b = None
  else:
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
  axes = [ax for ax in (a, b) if ax is not None]
  fig.patch.set_facecolor("white")
  for ax in axes:
    style(ax)
    ax.axhline(chance, color=MUTED, lw=1, ls=":")
    ax.plot(LAYERS, tools["probe_linear"], ls="--", color=INK, lw=1.3, zorder=2)
    ax.annotate("linear probe (supervised)", (LAYERS[-1], tools["probe_linear"].iloc[-1]),
                xytext=(8, 0), textcoords="offset points", va="center", fontsize=8.5,
                color=MUTED)
  a.annotate("chance 1.6%", (0, chance), xytext=(0, 4), textcoords="offset points",
             fontsize=8, color=MUTED)

  line(a, proto["img_all"], BLUE, "Image-slot injection (ours)", "o", lw=2.6, end_dy=6)
  line(a, tools["logit_lens"].tolist(), YELLOW, "Logit lens", "s", end_dy=-6)
  line(a, tools["latentlens"].tolist(), AQUA, "LatentLens", "D")
  line(a, proto["selfie_text"], ORANGE, "SelfIE (text slot)", "^")
  line(a, tools["patchscopes_pmi"].tolist(), MUTED, "Patchscopes (same layer)", "v", lw=1.4)
  a.set_ylabel("top-1 accuracy, object tokens", color=MUTED, fontsize=9)
  a.set_title("Training-free readouts of LLaVA-1.5 visual tokens", loc="left", fontsize=11,
              color=INK)

  if b is not None:
    line(b, proto["img_all"], BLUE, "Image slots", "o", lw=2.6, end_dy=-3)
    line(b, proto["img_all_multi"], AQUA, "Image slots + re-injection", "D", end_dy=8)
    line(b, proto["img_all_norm"], YELLOW, "Image slots + norm match", "s")
    line(b, proto["selfie_text"], ORANGE, "Text slot (SelfIE)", "^", end_dy=5)
    line(b, proto["selfie_text_norm"], MUTED, "Text slot + norm match", "v", lw=1.4, end_dy=-6)
    b.set_title("Where the state goes matters; its norm does not", loc="left", fontsize=11,
                color=INK)

  for ax in axes:
    ax.set_xlim(-1, 29)
    place_labels(ax)
  note = ("Fair 63-class subset. Injection rows: 200 COCO images, test split (70%), target "
          "layer chosen on dev split, PMI first-token scoring. Lens / probe rows: full "
          "benchmark (indicative, not matched).")
  if args.left_only:
    note = note.replace("first-token scoring. ", "first-token scoring.\n")
  fig.text(0.01, 0.01, note, fontsize=7.5, color=MUTED)
  if args.left_only:
    fig.tight_layout(rect=(0, 0.07, 0.8, 1))
  else:
    fig.tight_layout(rect=(0, 0.04, 0.86, 1), w_pad=9)
  fig.savefig(args.out, dpi=200, facecolor="white")
  print(f"saved {args.out}")


if __name__ == "__main__":
  main()
