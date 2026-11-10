r"""Does the model still need an object's visual tokens after layer K?

FastV (Chen et al., ECCV 2024) prunes half of the image tokens after layer 2 with little loss and
infers that image tokens matter little in deep layers. Pruning half at random usually keeps some
of the object's tokens. Here, after layer K, other positions can no longer attend to a chosen set
of image tokens (attention knockout from layer K + 1 on; the tokens' own states still run):

  object      every token the (dilated) object mask touches (as in causal_restore.py)
  random_k    the same number of image tokens the mask does not touch
  half        half of all image tokens at random (FastV's ratio; the share of object tokens it
              happens to remove is recorded)
  all_image   every image token

Per image (the benchmark's object, fair classes, main-object question) and K: P(class) relative
to the clean run (p / p_clean, as Neo et al. report blocking effects) and whether the top-1 is
still correct. Images count when the clean top-1 is correct.

The knockout edits the attention mask each decoder layer receives (a forward pre-hook). A
self-test runs first: with nothing blocked the hooked run must reproduce the clean log-probs.
Works for LLaVA-1.5 and Qwen2.5-VL with eager or SDPA attention (not FlashAttention).

  python run_object_pruning.py --coco_dir /path/to/coco --vtr_dir ./results/vtr_llava15 \
      --n_images 200 --after 0,2,4,8,12,16,20,24
    -> <vtr_dir>/pruning/{runs.csv, curve.csv}
"""

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from causal_restore import Runner, TokenGrid, choose_images, mask_object  # noqa: E402
from train_oracle import load_benchmark, load_model  # noqa: E402
from visual_object_identification import ann_to_mask, ensure_annotations, ensure_image  # noqa: E402

MODES = ("object", "random_k", "half", "all_image")


def parse_args(argv=None):
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--stage", default="all", choices=["all", "run", "summarize"])
  p.add_argument("--coco_dir", default=None)
  p.add_argument("--vtr_dir", default="./results/vtr_llava15")
  p.add_argument("--out_dir", default=None, help="default <vtr_dir>/pruning")
  p.add_argument("--model_name", default="llava-hf/llava-1.5-7b-hf")
  p.add_argument("--n_images", type=int, default=200)
  p.add_argument("--after", default="0,2,4,8,12,16,20,24",
                 help="K values: tokens are blocked in decoder layers K+1, K+2, ...")
  p.add_argument("--dilate", type=int, default=4)
  p.add_argument("--batch", type=int, default=8)
  p.add_argument("--min_pixels", type=int, default=56 * 56)
  p.add_argument("--max_pixels", type=int, default=12845056)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  return p.parse_args(argv)


def knockout_mask(mask, blocked, active, q_len, k_len, dtype, device):
  """The attention mask one decoder layer receives, with keys `blocked[r]` hidden from every
  other query position in the rows where `active[r]`.

  mask: what the layer was given (None = causal, bool = True attends, float = additive)."""
  b = len(blocked)
  if mask is None:
    mask = torch.ones(q_len, k_len, dtype=torch.bool, device=device).tril(k_len - q_len)
    mask = mask[None, None].expand(b, 1, -1, -1)
  mask = mask.clone() if mask.shape[0] == b else mask.expand(b, *mask.shape[1:]).clone()
  q_pos = torch.arange(k_len - q_len, k_len, device=device)
  for r, cols in enumerate(blocked):
    if not active[r] or not len(cols):
      continue
    cols = torch.as_tensor(cols, device=device)
    hide = q_pos[:, None] != cols[None, :]  # a token still attends to itself
    if mask.dtype == torch.bool:
      sub = mask[r, :, :, cols]
      mask[r, :, :, cols] = sub & ~hide
    else:
      sub = mask[r, :, :, cols]
      low = torch.full_like(sub, torch.finfo(sub.dtype).min)
      mask[r, :, :, cols] = torch.where(hide, torch.minimum(sub, low), sub)
  return mask


@contextmanager
def knockout(layers, blocked, after):
  """blocked[r]: sequence positions to hide in row r; after[r]: hide them in decoder layers with
  index > after[r] (None = never)."""
  def make(i):
    def pre(module, args, kwargs):
      active = [a is not None and i > a for a in after]
      if not any(active):
        return args, kwargs
      hs = args[0] if args else kwargs["hidden_states"]
      key = "attention_mask"
      if key in kwargs:
        mask = kwargs[key]
      elif len(args) > 1:
        mask, key = args[1], 1
      else:
        mask = None
      q_len = hs.shape[1]
      k_len = q_len if mask is None else mask.shape[-1]
      new = knockout_mask(mask, blocked, active, q_len, k_len, hs.dtype, hs.device)
      if key == 1:
        args = (args[0], new) + tuple(args[2:])
      else:
        kwargs[key] = new
      return args, kwargs
    return pre

  handles = [l.register_forward_pre_hook(make(i), with_kwargs=True) for i, l in enumerate(layers)]
  try:
    yield
  finally:
    for h in handles:
      h.remove()


def pruned_scores(runner, image, jobs, batch):
  """jobs: [(K, image-token indices)] -> class scores [len(jobs), n_kept]."""
  out = []
  for b in range(0, len(jobs), batch):
    chunk = jobs[b:b + batch]
    x, start = runner.inputs(image, len(chunk))
    with knockout(runner.layers, [start + np.asarray(t) for _, t in chunk],
                  [k for k, _ in chunk]):
      out.append(runner.logp(x))
  return np.concatenate(out)


def self_test(runner, image):
  clean, _ = runner.clean(image)
  hooked = pruned_scores(runner, image, [(0, np.array([], int)), (None, np.array([0]))], 2)
  ok = np.allclose(hooked, clean[None], atol=0.05)
  print(f"  self-test: nothing blocked reproduces the clean run: {ok} "
        f"(max |diff| {np.abs(hooked - clean[None]).max():.4f})")
  if not ok:
    raise SystemExit("the attention-mask hook changes the clean run: check the attention "
                     "implementation (eager or sdpa) and the transformers version")


def stage_run(args, out_dir, data, kept):
  chosen = choose_images(data["samples"], kept, args.n_images, args.seed)
  mt = load_model(args)
  mt.model.eval()
  family = getattr(mt, "family", "llava")
  if family == "qwen2.5-vl":
    mt.processor.set_pixel_bounds(args.min_pixels, args.max_pixels)
  impl = getattr(mt.model.config, "_attn_implementation", "?")
  print(f"  attention implementation: {impl}")
  if "flash" in str(impl):
    raise SystemExit("FlashAttention ignores custom masks: load with sdpa or eager")
  grid = TokenGrid(family, args.min_pixels, args.max_pixels)
  runner = Runner(mt, kept)
  after = [int(x) for x in args.after.split(",")]
  with open(ensure_annotations(args.coco_dir), encoding="utf-8") as f:
    anns = {a["id"]: a for a in json.load(f)["annotations"]}
  path = os.path.join(out_dir, "runs.csv")
  done = set(pd.read_csv(path)["image_id"]) if os.path.exists(path) else set()
  tested = False
  for k, s in enumerate(chosen):
    if s["image_id"] in done:
      continue
    t0 = time.time()
    image = Image.open(ensure_image(args.coco_dir, s["file_name"])).convert("RGB")
    mask = ann_to_mask(anns[s["ann_id"]], image.height, image.width)
    _, grown = mask_object(image, mask, args.dilate)
    cov = grid.coverage(grown)
    model_image = grid.model_image(image)
    if not tested:
      self_test(runner, model_image)
      tested = True
    obj = np.flatnonzero(cov > 0)
    rest = np.flatnonzero(cov == 0)
    every = np.arange(len(cov))
    rng = np.random.RandomState(args.seed + int(s["image_id"]))
    sets = {"object": obj,
            "random_k": np.sort(rng.choice(rest, min(len(obj), len(rest)), replace=False)),
            "half": np.sort(rng.choice(every, len(every) // 2, replace=False)),
            "all_image": every}
    y = kept.index(s["class_name"])
    clean, _ = runner.clean(model_image)
    specs = [(m, K) for K in after for m in MODES]
    scores = pruned_scores(runner, model_image, [(K, sets[m]) for m, K in specs], args.batch)
    p_clean = float(np.exp(clean[y]))
    rows = [{"image_id": s["image_id"], "class_name": s["class_name"], "mode": m, "after": K,
             "n_blocked": len(sets[m]), "n_object": len(obj),
             "object_share_blocked": float(np.isin(obj, sets[m]).mean()) if len(obj) else np.nan,
             "p_clean": p_clean, "top1_clean": bool(clean.argmax() == y),
             "p": float(np.exp(sc[y])), "top1": bool(sc.argmax() == y)}
            for (m, K), sc in zip(specs, scores)]
    pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    print(f"  [{k + 1}/{len(chosen)}] {s['class_name']}: p clean {p_clean:.2f}, {len(obj)} object "
          f"tokens ({time.time() - t0:.0f}s)", flush=True)


def stage_summarize(out_dir):
  runs = pd.read_csv(os.path.join(out_dir, "runs.csv"))
  runs = runs[runs["top1_clean"]]
  runs["relative"] = runs["p"] / runs["p_clean"]
  curve = runs.groupby(["mode", "after"]).agg(relative=("relative", "mean"),
                                             top1=("top1", "mean"),
                                             object_share=("object_share_blocked", "mean"),
                                             n_images=("image_id", "nunique")).reset_index()
  curve.to_csv(os.path.join(out_dir, "curve.csv"), index=False)
  pd.set_option("display.width", 200)
  print(f"\n{runs['image_id'].nunique()} images with a correct clean answer.")
  for col, what in (("relative", "P(class) / P_clean"), ("top1", "top-1 still correct")):
    print(f"\n{what}, by K (tokens hidden from layer K+1 on):")
    print(curve.pivot(index="mode", columns="after", values=col).reindex(MODES).round(2)
          .to_string())
  print("\nShare of the object's tokens that 'half' happens to remove:")
  print(curve[curve["mode"] == "half"].set_index("after")["object_share"].round(2).to_string())
  print(f"\nAll outputs in {out_dir}")


def main(argv=None):
  args = parse_args(argv)
  out_dir = args.out_dir or os.path.join(args.vtr_dir, "pruning")
  os.makedirs(out_dir, exist_ok=True)
  if args.stage in ("all", "run"):
    if not args.coco_dir:
      raise SystemExit("--coco_dir is needed to run")
    data, kept = load_benchmark(args.vtr_dir)
    stage_run(args, out_dir, data, kept)
  if args.stage in ("all", "summarize"):
    stage_summarize(out_dir)


if __name__ == "__main__":
  main()
