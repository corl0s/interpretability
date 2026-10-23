r"""How often is the oracle's teacher right? LLaVA's answer about one token's region, vs COCO.

The visual activation oracle (train_oracle.py) learns to reproduce LLaVA's own answer about the
region of a visual token, so its accuracy is capped by how often that answer names the token's
object. This script asks LLaVA about the benchmark object tokens (the train_oracle eval rows) with
several ways of pointing at the region (vtr.oracle.make_view), and scores each against COCO:

  full     the whole image, "What is the main object?"     (no pointing: the main-object baseline)
  boxK     red box around the KxK patch block, "What is the object inside the red box?"
           (box3 is how the pilot's training labels were made)
  cropK    only the KxK patch block, enlarged to 336x336, "What is the main object?"
  maskK    the whole image, everything outside the KxK block greyed out, "What is the main object?"

Two scores per view, on object tokens of the test images (same dev/test split as the oracle eval):
  closed   first-token top-1 over the fair classes (raw log-prob, nothing generated): comparable
           to the oracle's top-1 in train_oracle.py's eval table
  free     the generated answer (greedy, <= 6 tokens) names the class or a synonym

  python measure_teacher.py --coco_dir /projectnb/mlresearch/vishnuav/coco \
      --vtr_dir ./results/vtr_llava15
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_object_identification import bootstrap_by_image  # noqa: E402
from run_ps_sweep import select_rows  # noqa: E402
from train_oracle import load_benchmark, load_model  # noqa: E402
from visual_object_identification import ensure_image, mentions  # noqa: E402
from vtr import evaluate as E  # noqa: E402
from vtr.oracle import GREY, ask, block, class_scores, first_tokens, make_view  # noqa: E402,F401
from vtr.oracle import square_crop  # noqa: E402
from vtr.readouts import class_forms  # noqa: E402


def parse_args(argv=None):
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--coco_dir", required=True)
  p.add_argument("--vtr_dir", default="./results/vtr_llava15")
  p.add_argument("--out_dir", default=None, help="default: <vtr_dir>/oracle_pilot/teacher")
  p.add_argument("--model_name", default="llava-hf/llava-1.5-7b-hf")
  p.add_argument("--views", default="full,box3,box5,crop3,crop5,crop7,mask5,mask7")
  p.add_argument("--eval_images", type=int, default=200)
  p.add_argument("--batch", type=int, default=16)
  p.add_argument("--n_boot", type=int, default=500)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  return p.parse_args(argv)


def main(argv=None, mt=None):
  args = parse_args(argv)
  out_dir = args.out_dir or os.path.join(args.vtr_dir, "oracle_pilot", "teacher")
  os.makedirs(out_dir, exist_ok=True)
  data, kept = load_benchmark(args.vtr_dir)
  class_names = data["class_names"]
  files = {s["image_id"]: s["file_name"] for s in data["samples"]}
  meta = torch.load(os.path.join(args.vtr_dir, "states.pt"), weights_only=False)["meta"]
  rows = meta.iloc[select_rows(meta, args.eval_images, args.seed)].reset_index(drop=True)

  # Object rows of fair classes, and the oracle eval's dev/test split (by image).
  rows = rows[(rows["kind"] == "object") & rows["class_name"].isin(kept)].reset_index(drop=True)
  images = rows["image_id"].drop_duplicates().sample(frac=1.0, random_state=args.seed).to_numpy()
  dev = set(images[:int(round(0.3 * len(images)))])
  rows["split"] = np.where(rows["image_id"].isin(dev), "dev", "test")
  print(f"{len(rows)} object rows ({(rows.split == 'test').sum()} test) "
        f"from {rows.image_id.nunique()} images")

  if mt is None:
    mt = load_model(args)
  forms = class_forms(kept, synonyms=False)
  cls, ids = first_tokens(mt.tokenizer, kept, forms)
  syn = class_forms(kept, synonyms=True)
  lab = rows["class_name"].map({c: i for i, c in enumerate(kept)}).to_numpy()

  cache = {}
  for view in args.views.split(","):
    path = os.path.join(out_dir, f"{view}.csv")
    if os.path.exists(path):
      cache[view] = pd.read_csv(path, keep_default_na=False)
      continue
    t0 = time.time()
    answers, scores = [], []
    for b in range(0, len(rows), args.batch):
      chunk = rows.iloc[b:b + args.batch]
      views = []
      for _, r in chunk.iterrows():
        image = square_crop(Image.open(ensure_image(args.coco_dir, files[r.image_id]))
                            .convert("RGB"))
        views.append(make_view(image, r.token, view))
      a, logp = ask(mt, [v[0] for v in views], views[0][1])
      answers += a
      scores.append(class_scores(logp, cls, ids, len(kept)))
    scores = np.concatenate(scores)
    df = rows[["image_id", "token", "class_name", "split"]].copy()
    df["answer"] = answers
    df["closed_pred"] = [kept[i] for i in scores.argmax(1)]
    df["closed"] = (E.ranks(scores, lab) == 1).astype(int)
    df["free"] = [int(any(mentions(a, f) for f in syn[c])) for a, c in zip(answers, df.class_name)]
    df.to_csv(path, index=False)
    cache[view] = df
    print(f"  {view}: {len(df)} rows ({time.time() - t0:.0f}s); e.g. {answers[:6]}", flush=True)

  out = []
  for view, df in cache.items():
    test = df[df.split == "test"]
    rec = {"view": view, "n": len(test)}
    for metric in ("closed", "free"):
      b = bootstrap_by_image(test.assign(layer=0), metric, args.n_boot, args.seed).iloc[0]
      rec[metric] = b["mean"]
      rec[f"{metric}_ci"] = f"{b['ci_lo']:.3f}-{b['ci_hi']:.3f}"
    if "full" in cache and view != "full":
      rec["same_as_full"] = float((df.answer.values == cache["full"].answer.values).mean())
    out.append(rec)
  table = pd.DataFrame(out).set_index("view")
  table.to_csv(os.path.join(out_dir, "teacher_summary.csv"))
  pd.set_option("display.width", 200)
  print("\nTeacher accuracy, object tokens, test images (oracle pilot: 0.478 mean, "
        "0.526 at s28; probe 0.604):")
  print(table.round(3).to_string())
  print("  closed = first-token top-1 over the fair classes; free = generated answer names the "
        "class;\n  same_as_full = answer identical to the whole-image answer (pointing ignored)")
  print(f"\nAll outputs in {out_dir}")


if __name__ == "__main__":
  main()
