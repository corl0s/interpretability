r"""Post-processing for visual_object_identification.py results. No GPU required.

Three jobs, all on the CSVs the experiment already wrote:

1. **Cluster bootstrap by image.** 4 patches per image share one object, so rows are not
   independent; the independent unit is the image. All confidence intervals resample images
   with replacement.

2. **Lenient rescoring.** Exact-lemma containment scores `airplane -> "Aircraft with a single
   set of wings"` as wrong. A curated surface-form table (below) adds synonyms, hypernyms and
   common hyponyms. Both the exact and the lenient number are reported side by side -- the
   lenient one is hand-curated and therefore the weaker of the two; never quote it alone.

3. **Ambiguity diagnostics.** Classes whose names are frequent English words ("person",
   "train", "orange") can be matched by generic text with no visual information. Rather than
   judging ambiguity by eye, this measures each class's false-positive rate in the *control*
   conditions (outside-object patches and norm-matched random vectors) and reports the headline
   restricted to classes whose control FP rate is zero.

Also splits the logit-lens comparison by single- vs multi-word class names, because the
logit-lens criterion only needs the first sub-token ("traffic" for "traffic light") while
containment needs the whole string.

  python analyze_object_identification.py --results_dir ./results/object_identification
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd

# Curated additional surface forms per COCO class: synonyms, hypernyms and frequent hyponyms.
# Deliberately conservative -- a form is listed only if a generation containing it would be
# accepted as correct by a human marker looking at the patch.
SURFACE_FORMS = {
    "airplane": ["aircraft", "plane", "jet", "airliner"],
    "bicycle": ["bike", "cycle"],
    "motorcycle": ["motorbike", "bike"],
    "car": ["automobile", "vehicle", "sedan"],
    "truck": ["lorry", "pickup"],
    "bus": ["coach", "omnibus"],
    "train": ["locomotive", "railway", "railroad"],
    "boat": ["ship", "vessel", "sailboat", "canoe"],
    "dog": ["puppy", "canine", "retriever", "terrier", "labrador"],
    "cat": ["kitten", "feline", "tabby"],
    "horse": ["pony", "stallion", "mare"],
    "sheep": ["lamb", "ewe", "ram"],
    "cow": ["cattle", "bull", "calf", "ox"],
    "bird": ["seagull", "pigeon", "duck", "sparrow"],
    "bear": ["grizzly", "panda"],
    "person": ["man", "woman", "human", "people", "child", "boy", "girl"],
    "tv": ["television", "monitor", "screen"],
    "laptop": ["notebook computer", "computer"],
    "cell phone": ["mobile phone", "smartphone", "phone"],
    "remote": ["remote control"],
    "couch": ["sofa", "settee"],
    "potted plant": ["houseplant", "plant", "pot plant"],
    "dining table": ["table"],
    "refrigerator": ["fridge", "freezer"],
    "microwave": ["microwave oven"],
    "oven": ["stove", "cooker"],
    "sink": ["basin", "washbasin"],
    "toilet": ["lavatory", "wc"],
    "cup": ["mug", "glass"],
    "wine glass": ["glass", "goblet"],
    "bottle": ["flask"],
    "bowl": ["dish"],
    "fork": ["cutlery"],
    "knife": ["blade", "cutlery"],
    "spoon": ["cutlery"],
    "teddy bear": ["stuffed animal", "plush toy", "soft toy"],
    "traffic light": ["traffic signal", "stoplight"],
    "stop sign": ["road sign", "traffic sign"],
    "fire hydrant": ["hydrant"],
    "parking meter": ["meter"],
    "sports ball": ["ball", "football", "soccer ball", "basketball"],
    "baseball bat": ["bat"],
    "baseball glove": ["glove", "mitt"],
    "tennis racket": ["racket", "racquet"],
    "surfboard": ["board"],
    "skateboard": ["board"],
    "snowboard": ["board"],
    "umbrella": ["parasol"],
    "handbag": ["purse", "bag"],
    "suitcase": ["luggage", "case"],
    "backpack": ["rucksack", "bag"],
    "hot dog": ["frankfurter", "sausage"],
    "donut": ["doughnut"],
    "couch ": ["sofa"],
    "hair drier": ["hairdryer", "hair dryer"],
    "toothbrush": ["brush"],
}


def mentions_any(text, names):
  """True if any surface form occurs as a whole word (simple plural allowed)."""
  if not isinstance(text, str):
    return False
  low = text.lower()
  for name in names:
    if re.search(r"\b" + re.escape(name.lower()) + r"(e?s)?\b", low):
      return True
  return False


def add_lenient(df):
  """Add a `correct_lenient` column using the curated surface forms."""
  forms = {c: [c] + SURFACE_FORMS.get(c, []) for c in df["class_name"].unique()}
  df = df.copy()
  df["correct_lenient"] = [
      mentions_any(g, forms[c]) for g, c in zip(df["generation"], df["class_name"])
  ]
  return df


def class_control_fp(df, score_col):
  """False-positive rate per class over the control conditions (outside + random)."""
  ctrl = df[df["kind"].isin(["outside", "random"])]
  return ctrl.groupby("class_name")[score_col].mean()


def bootstrap_by_image(df, score_col, n_boot, seed):
  """Per-layer mean with a 95% CI, resampling IMAGES (not rows) with replacement."""
  rng = np.random.RandomState(seed)
  images = df["image_id"].unique()
  by_image_layer = (df.groupby(["image_id", "layer"])[score_col]
                    .agg(["sum", "count"]).reset_index())
  layers = sorted(df["layer"].unique())
  sums = by_image_layer.pivot(index="image_id", columns="layer", values="sum").fillna(0)
  counts = by_image_layer.pivot(index="image_id", columns="layer", values="count").fillna(0)
  sums, counts = sums.loc[images].to_numpy(), counts.loc[images].to_numpy()

  point = np.divide(sums.sum(0), counts.sum(0),
                    out=np.zeros(len(layers)), where=counts.sum(0) > 0)
  draws = np.empty((n_boot, len(layers)))
  for b in range(n_boot):
    idx = rng.randint(0, len(images), len(images))
    s, c = sums[idx].sum(0), counts[idx].sum(0)
    draws[b] = np.divide(s, c, out=np.zeros(len(layers)), where=c > 0)
  lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
  return pd.DataFrame({"layer": layers, "mean": point, "ci_lo": lo, "ci_hi": hi})


def summarize_condition(df, score_col, n_boot, seed, label):
  out = bootstrap_by_image(df, score_col, n_boot, seed)
  out["condition"] = label
  return out


def main():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--results_dir", default="./results/object_identification")
  p.add_argument("--prompt", default="entity")
  p.add_argument("--n_boot", type=int, default=2000)
  p.add_argument("--seed", type=int, default=0)
  args = p.parse_args()

  ps = pd.read_csv(os.path.join(args.results_dir, "patchscopes.csv"))
  ps = ps[ps["prompt_key"] == args.prompt]
  ps = add_lenient(ps)
  lens = pd.read_csv(os.path.join(args.results_dir, "logit_lens.csv"))

  # ---------------------------------------------------------------- ambiguity diagnostics
  fp_exact = class_control_fp(ps, "correct")
  fp_lenient = class_control_fp(ps, "correct_lenient")
  clean_classes = sorted(fp_exact[fp_exact == 0].index)
  print(f"classes with zero control false-positives: {len(clean_classes)} "
        f"of {ps['class_name'].nunique()}")
  worst = fp_exact.sort_values(ascending=False).head(10)
  print("\nhighest control false-positive rates (candidate ambiguous class names):")
  print(worst.to_string())
  pd.DataFrame({"control_fp_exact": fp_exact,
                "control_fp_lenient": fp_lenient}).to_csv(
                    os.path.join(args.results_dir, "class_control_fp.csv"))

  # ---------------------------------------------------------------- bootstrap summaries
  frames = []
  for subset_name, subset in (("all", ps), ("clean", ps[ps["class_name"].isin(clean_classes)])):
    for kind in ("object", "outside", "random"):
      sub = subset[subset["kind"] == kind]
      if not len(sub):
        continue
      for score_col, tag in (("correct", "exact"), ("correct_lenient", "lenient")):
        frame = summarize_condition(sub, score_col, args.n_boot, args.seed,
                                    f"ps_{kind}_{tag}")
        frame["subset"] = subset_name
        frames.append(frame)

  lens_all = lens.rename(columns={"top1_correct": "correct"})
  for subset_name, subset in (("all", lens_all),
                              ("clean", lens_all[lens_all["class_name"].isin(clean_classes)])):
    for col, tag in (("correct", "top1"), ("top5_correct", "top5")):
      frame = summarize_condition(subset, col, args.n_boot, args.seed, f"logit_lens_{tag}")
      frame["subset"] = subset_name
      frames.append(frame)

  summary = pd.concat(frames, ignore_index=True)
  summary.to_csv(os.path.join(args.results_dir, "bootstrap_by_layer.csv"), index=False)

  # ---------------------------------------------------------------- multi-word split
  lens_all["multiword"] = lens_all["class_name"].str.contains(" ")
  split = (lens_all.groupby(["layer", "multiword"])["correct"].mean()
           .unstack().rename(columns={False: "single_word", True: "multi_word"}))
  split.to_csv(os.path.join(args.results_dir, "logit_lens_wordsplit.csv"))

  # ---------------------------------------------------------------- headline
  def at(condition, subset, layer):
    row = summary[(summary.condition == condition) & (summary.subset == subset)
                  & (summary.layer == layer)]
    if not len(row):
      return None
    r = row.iloc[0]
    return {"mean": round(float(r["mean"]), 4),
            "ci": [round(float(r["ci_lo"]), 4), round(float(r["ci_hi"]), 4)]}

  obj = summary[(summary.condition == "ps_object_exact") & (summary.subset == "clean")]
  early = obj[obj.layer <= 6]
  late = obj[obj.layer >= 26]
  best_early = int(early.loc[early["mean"].idxmax(), "layer"]) if len(early) else None
  best_late = int(late.loc[late["mean"].idxmax(), "layer"]) if len(late) else None

  headline = {
      "prompt": args.prompt,
      "n_images": int(ps["image_id"].nunique()),
      "n_classes": int(ps["class_name"].nunique()),
      "n_clean_classes": len(clean_classes),
      "early_peak_layer": best_early,
      "late_peak_layer": best_late,
      "clean_subset": {
          "ps_object_exact_early": at("ps_object_exact", "clean", best_early),
          "ps_object_lenient_early": at("ps_object_lenient", "clean", best_early),
          "ps_outside_exact_early": at("ps_outside_exact", "clean", best_early),
          "ps_random_exact_early": at("ps_random_exact", "clean", best_early),
          "logit_lens_top1_early": at("logit_lens_top1", "clean", best_early),
          "ps_object_exact_late": at("ps_object_exact", "clean", best_late),
          "logit_lens_top1_late": at("logit_lens_top1", "clean", best_late),
      },
      "logit_lens_wordsplit_best_layer": {
          "single_word": round(float(split["single_word"].max()), 4)
                         if "single_word" in split else None,
          "multi_word": round(float(split["multi_word"].max()), 4)
                        if "multi_word" in split else None,
      },
  }
  with open(os.path.join(args.results_dir, "headline_rescored.json"), "w",
            encoding="utf-8") as f:
    json.dump(headline, f, indent=2)

  print("\n=== clean-subset, exact scoring, with image-clustered 95% CIs ===")
  wide = summary[summary.subset == "clean"].pivot_table(
      index="layer", columns="condition", values=["mean", "ci_lo", "ci_hi"])
  show = pd.DataFrame({
      "ps_object": wide[("mean", "ps_object_exact")].round(3),
      "ps_object_lo": wide[("ci_lo", "ps_object_exact")].round(3),
      "ps_object_hi": wide[("ci_hi", "ps_object_exact")].round(3),
      "ps_object_lenient": wide[("mean", "ps_object_lenient")].round(3),
      "ps_outside": wide[("mean", "ps_outside_exact")].round(3),
      "ps_random": wide[("mean", "ps_random_exact")].round(3),
      "logit_lens": wide[("mean", "logit_lens_top1")].round(3),
  })
  print(show.to_string())
  print("\n=== headline ===")
  print(json.dumps(headline, indent=2))

  # ---------------------------------------------------------------- plot
  try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for cond, colour, style in (("ps_object_exact", "C0", "-"),
                                ("logit_lens_top1", "C1", "-"),
                                ("ps_outside_exact", "C7", "--"),
                                ("ps_random_exact", "C8", ":")):
      sub = summary[(summary.condition == cond) & (summary.subset == "clean")]
      if not len(sub):
        continue
      ax.plot(sub["layer"], sub["mean"], style, color=colour, marker="o", ms=3, label=cond)
      ax.fill_between(sub["layer"], sub["ci_lo"], sub["ci_hi"], color=colour, alpha=0.15)
    ax.set_xlabel("LLM backbone layer")
    ax.set_ylabel("object identification accuracy")
    ax.set_title("Patchscopes vs. logit lens on visual patch tokens\n"
                 "(unambiguous classes, image-clustered 95% CI)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.results_dir, "readout_comparison.png"), dpi=150)
    print(f"\nwrote {os.path.join(args.results_dir, 'readout_comparison.png')}")
  except ImportError:
    pass


if __name__ == "__main__":
  main()
