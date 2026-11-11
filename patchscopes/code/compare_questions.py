"""Compare the causal test across questions (job_causal_questions.sh outputs, CPU, no model).

For each question: valid images, recovery from restoring the object's tokens at each layer, the
last layer where mean recovery is still >= 0.5, and P(decodes | used) / P(decodes | not used)
for a few readouts. The visual states do not depend on the question (the text follows the
image), so only the "used" side can move.

  python compare_questions.py --vtr_dir ./results/vtr_llava15 \
      --dirs main=causal_fixed,photo=causal_q_photo,describe=causal_q_describe,yesno=causal_q_yesno
"""

import argparse
import os

import pandas as pd

READOUTS = ["logit_lens", "image_slot_pmi", "probe_linear"]


def main():
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--vtr_dir", default="./results/vtr_llava15")
  p.add_argument("--dirs", required=True, help="name=subdir,... under --vtr_dir")
  p.add_argument("--min_gap", type=float, default=0.2)
  args = p.parse_args()

  curves, cases, info = {}, {}, []
  for item in args.dirs.split(","):
    name, sub = item.split("=")
    d = os.path.join(args.vtr_dir, sub)
    if not os.path.exists(os.path.join(d, "curve.csv")):
      print(f"{name}: no curve.csv in {d} (not finished?)")
      continue
    images = pd.read_csv(os.path.join(d, "images.csv"))
    valid = images["top1_clean"] & (images["p_clean"] - images["p_masked"] >= args.min_gap)
    curve = pd.read_csv(os.path.join(d, "curve.csv"))
    curve = curve[curve["mode"] == "object"].set_index("layer")["recovery_mean"]
    curves[name] = curve
    used = curve[curve >= 0.5].index
    info.append({"question": name, "images": len(images), "valid": int(valid.sum()),
                 "last layer recovery>=0.5": int(used.max()) if len(used) else None})
    c = pd.read_csv(os.path.join(d, "cases.csv"), index_col=0)
    cases[name] = c.loc[[r for r in READOUTS if r in c.index], ["hit_used", "hit_unused"]]

  pd.set_option("display.width", 200)
  print(pd.DataFrame(info).to_string(index=False))
  print("\nMean recovery by restore layer (object tokens):")
  print(pd.DataFrame(curves).T.round(2).to_string())
  print("\nP(decodes | used) / P(decodes | not used):")
  print(pd.concat(cases, names=["question", "readout"]).round(2).to_string())


if __name__ == "__main__":
  main()
