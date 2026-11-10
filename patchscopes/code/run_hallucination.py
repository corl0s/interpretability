r"""Hallucination detection with visual-token readouts (D1): logit lens vs image-slot injection.

Jiang et al. (ICLR 2025, 2410.02762) score each object a VLM mentions by its "internal
confidence": the highest logit-lens probability of the object's token over ALL layers and image
tokens; real objects score higher than hallucinated ones. Our causal test says the model stops
relying on the object's tokens around the middle of the stack, while the logit lens reads them
well only late. This script asks which readout, at which layers, separates real from hallucinated
mentions best.

  1. caption --n_images COCO val2017 images (not in the benchmark) with --prompt, greedy
  2. find the COCO classes the caption mentions (class names + the project's unambiguous
     synonyms, SURFACE_FORMS; a match inside a longer match of another class is dropped). A
     mention is real if the image's annotations contain the class, hallucinated otherwise (the
     CHAIR idea; our synonym list, not CHAIR's)
  3. score every mentioned class:
       out_prob      probability of the mention's first token when the model generated it
       lens_all      max logit-lens probability over all layers and image tokens (Jiang et al.)
       lens_used     the same over layers <= --used_until (where restoring object tokens still
                     brings the answer back; causal_restore.py)
       lens_late     the same over layers > --used_until
       slot_s<L>     image-slot PMI, max over image tokens (every --slot_stride-th row and
                     column) of source layer L, target from <vtr_dir>/injection_proto
       slot_max      max over the slot_s<L>
  4. AUROC (real vs hallucinated; higher = real) and AP with hallucinated as the positive class
     (score negated), with image-clustered bootstrap CIs

  python run_hallucination.py --coco_dir /path/to/coco --vtr_dir ./results/vtr_llava15 \
      --n_images 500
    -> <vtr_dir>/hallucination/{captions.csv, mentions.csv, metrics.csv}

Resumable per image. --stage summarize only re-reads mentions.csv.
"""

import argparse
import json
import os
import re
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_vtr  # noqa: E402
from visual_object_identification import (class_token_ids, ensure_annotations,  # noqa: E402
                                          ensure_image, get_decoder)
from vtr.injection import InjectionReadout  # noqa: E402
from vtr.models import chat_inputs  # noqa: E402
from vtr.readouts import class_forms  # noqa: E402

# The captioning prompt of Jiang et al., "Devils in Middle Layers" (CVPR 2025, Sec. 3.2).
PROMPT = "Please help me describe the image in detail."


def parse_args(argv=None):
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--stage", default="all", choices=["all", "run", "summarize"])
  p.add_argument("--coco_dir", default=None)
  p.add_argument("--vtr_dir", default="./results/vtr_llava15")
  p.add_argument("--out_dir", default=None, help="default <vtr_dir>/hallucination")
  p.add_argument("--model_name", default="llava-hf/llava-1.5-7b-hf")
  p.add_argument("--n_images", type=int, default=500)
  p.add_argument("--prompt", default=PROMPT)
  p.add_argument("--max_new_tokens", type=int, default=256)
  p.add_argument("--used_until", type=int, default=18,
                 help="last layer counted as 'used' (LLaVA-1.5-7B: recovery 0.50 at layer 18)")
  p.add_argument("--slot_layers", default="0,8,16")
  p.add_argument("--slot_target", default="dev", help="as in run_vtr.py")
  p.add_argument("--slot_stride", type=int, default=2)
  p.add_argument("--slot_rows_per_batch", type=int, default=32)
  p.add_argument("--min_pixels", type=int, default=56 * 56)
  p.add_argument("--max_pixels", type=int, default=12845056)
  p.add_argument("--n_boot", type=int, default=1000)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  return p.parse_args(argv)


# ---------------------------------------------------------------------------------- mentions


def find_mentions(text, forms):
  """{class: char start of its first mention}. forms: {class: [surface forms]}. A match that
  lies inside a longer match of another class (e.g. "dog" in "hot dog") is dropped."""
  spans = []
  for c, fs in forms.items():
    for f in fs:
      for m in re.finditer(r"\b" + re.escape(f.lower()) + r"(e?s)?\b", text.lower()):
        spans.append((m.start(), m.end(), c))
  out = {}
  for s, e, c in spans:
    inside = any(s2 <= s and e <= e2 and (e2 - s2) > (e - s) and c2 != c
                 for s2, e2, c2 in spans)
    if not inside and (c not in out or s < out[c]):
      out[c] = s
  return out


def token_at(tokenizer, ids, char):
  """Index of the generated token whose decoded text covers character `char`."""
  for i in range(len(ids)):
    if len(tokenizer.decode(ids[:i + 1], skip_special_tokens=True)) > char:
      return i
  return len(ids) - 1


# ---------------------------------------------------------------------------------- model


@torch.no_grad()
def caption(mt, image, args):
  x, start, n = chat_inputs(mt, [image], args.prompt)
  tok = mt.tokenizer
  pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
  layers = get_decoder(mt).layers
  got = {}

  def hook(k):
    def fn(module, inp, out):
      if k not in got:  # the prefill pass only
        got[k] = (out[0] if isinstance(out, tuple) else out)[0, start:start + n].detach()
    return fn

  handles = [l.register_forward_hook(hook(k)) for k, l in enumerate(layers)]
  try:
    out = mt.model.generate(**x, max_new_tokens=args.max_new_tokens, do_sample=False,
                            pad_token_id=pad, output_scores=True, return_dict_in_generate=True)
  finally:
    for h in handles:
      h.remove()
  new = out.sequences[0, x["input_ids"].shape[1]:].tolist()
  probs = [float(s[0].float().softmax(-1)[t]) for s, t in zip(out.scores, new)]
  states = torch.stack([got[k] for k in range(len(layers))])  # [L, n, d]
  return tok.decode(new, skip_special_tokens=True), new, probs, states


class LensConfidence:
  """Max logit-lens probability of a class's first sub-tokens over the image tokens, per layer."""

  def __init__(self, mt, class_names, forms):
    self.norm, self.head = get_decoder(mt).norm, mt.model.lm_head
    self.ids = [sorted({t for f in forms[c] for t in class_token_ids(mt.tokenizer, f)})
                for c in class_names]

  @torch.no_grad()
  def __call__(self, states):
    """states [L, n, d] -> [L, n_classes]."""
    dtype = next(self.head.parameters()).dtype
    out = []
    for h in states:
      p = self.head(self.norm(h.to(dtype))).float().softmax(-1)  # [n, V]
      out.append(torch.stack([p[:, ids].max() for ids in self.ids]).cpu())
    return torch.stack(out).numpy()


def stage_run(args, out_dir):
  with open(os.path.join(args.vtr_dir, "samples.json"), encoding="utf-8") as f:
    bench = json.load(f)
  class_names = bench["class_names"]
  used_ids = {s["image_id"] for s in bench["samples"]}
  with open(ensure_annotations(args.coco_dir), encoding="utf-8") as f:
    coco = json.load(f)
  cat = {c["id"]: c["name"] for c in coco["categories"]}
  present = {}
  for a in coco["annotations"]:
    present.setdefault(a["image_id"], set()).add(cat[a["category_id"]])
  images = sorted((i for i in coco["images"] if i["id"] in present and i["id"] not in used_ids),
                  key=lambda i: i["id"])
  rng = np.random.RandomState(args.seed)
  images = [images[i] for i in rng.permutation(len(images))[:args.n_images]]

  mt = run_vtr.load_model(args)
  forms_syn = class_forms(class_names, synonyms=True)
  forms = class_forms(class_names, synonyms=False)
  lens = LensConfidence(mt, class_names, forms_syn)
  slot_layers = [int(x) for x in args.slot_layers.split(",") if x]
  targets = run_vtr.slot_targets(SimpleNamespace(slot_target=args.slot_target,
                                                 out_dir=args.vtr_dir), slot_layers)
  print(f"image-slot target per source layer: {targets}")
  ro = InjectionReadout(mt, class_names, forms, "image", args.device,
                        rows_per_batch=args.slot_rows_per_batch)

  cap_path, men_path = (os.path.join(out_dir, f) for f in ("captions.csv", "mentions.csv"))
  done = set(pd.read_csv(cap_path)["image_id"]) if os.path.exists(cap_path) else set()
  for k, info in enumerate(images):
    if info["id"] in done:
      continue
    t0 = time.time()
    image = Image.open(ensure_image(args.coco_dir, info["file_name"])).convert("RGB")
    text, ids, probs, states = caption(mt, image, args)
    found = find_mentions(text, forms_syn)
    rows = []
    if found:
      conf = lens(states)  # [L, C]
      side = int(round(states.shape[1] ** 0.5))
      grid = [r * side + c for r in range(0, side, args.slot_stride)
              for c in range(0, side, args.slot_stride)] if side * side == states.shape[1] \
          else list(range(0, states.shape[1], args.slot_stride ** 2))
      slot = {}
      for l in slot_layers:
        h = states[l, grid].float()
        slot[l] = ro.scores(h, targets[l], [ro.slots("all")] * len(h))[1].max(0)  # [C]
      L = conf.shape[0]
      used = [l for l in range(L) if l <= args.used_until]
      late = [l for l in range(L) if l > args.used_until]
      for c, char in found.items():
        ci = class_names.index(c)
        row = {"image_id": info["id"], "class_name": c, "real": c in present[info["id"]],
               "out_prob": probs[token_at(mt.tokenizer, ids, char)],
               "lens_all": float(conf[:, ci].max()), "lens_used": float(conf[used, ci].max()),
               "lens_late": float(conf[late, ci].max()) if late else np.nan,
               **{f"lens_L{l}": float(conf[l, ci]) for l in range(L)},
               **{f"slot_s{l}": float(v[ci]) for l, v in slot.items()}}
        row["slot_max"] = max(row[f"slot_s{l}"] for l in slot_layers) if slot_layers else np.nan
        rows.append(row)
    if rows:
      pd.DataFrame(rows).to_csv(men_path, mode="a", header=not os.path.exists(men_path),
                                index=False)
    pd.DataFrame([{"image_id": info["id"], "caption": text, "n_mentions": len(rows),
                   "n_hallucinated": sum(not r["real"] for r in rows)}]).to_csv(
        cap_path, mode="a", header=not os.path.exists(cap_path), index=False)
    print(f"  [{k + 1}/{len(images)}] {len(rows)} mentions, "
          f"{sum(not r['real'] for r in rows)} hallucinated ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------------------------- summary


def metrics(m, col, n_boot, seed):
  from sklearn.metrics import average_precision_score, roc_auc_score
  d = m[["image_id", "real", col]].dropna()
  y, s = d["real"].astype(int).to_numpy(), d[col].to_numpy()
  if len(set(y)) < 2:
    return None
  out = {"score": col, "auroc": roc_auc_score(y, s), "ap_hallucinated": average_precision_score(
      1 - y, -s), "n": len(d), "n_hallucinated": int((y == 0).sum())}
  groups = d.groupby("image_id").indices
  ids = list(groups)
  rng = np.random.RandomState(seed)
  boot = []
  for _ in range(n_boot):
    idx = np.concatenate([groups[ids[i]] for i in rng.randint(0, len(ids), len(ids))])
    if len(set(y[idx])) == 2:
      boot.append(roc_auc_score(y[idx], s[idx]))
  out["auroc_lo"], out["auroc_hi"] = np.percentile(boot, [2.5, 97.5])
  return out


def stage_summarize(args, out_dir):
  m = pd.read_csv(os.path.join(out_dir, "mentions.csv"))
  caps = pd.read_csv(os.path.join(out_dir, "captions.csv"))
  print(f"\n{len(caps)} captions, {len(m)} mentions, {(~m['real']).sum()} hallucinated "
        f"(CHAIR_i-style rate {(~m['real']).mean():.3f}; captions with a hallucination "
        f"{(caps['n_hallucinated'] > 0).mean():.3f})")
  cols = ["out_prob", "lens_all", "lens_used", "lens_late", "slot_max"] + \
      sorted([c for c in m.columns if c.startswith("slot_s")], key=lambda c: int(c[6:])) + \
      sorted([c for c in m.columns if c.startswith("lens_L")], key=lambda c: int(c[6:]))
  rows = [r for r in (metrics(m, c, args.n_boot, args.seed) for c in cols) if r]
  tab = pd.DataFrame(rows)
  tab.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
  pd.set_option("display.width", 200)
  print("\nReal vs hallucinated mentions (AUROC: higher score = real; AP: hallucinated as the "
        "positive class):")
  print(tab.round(3).to_string(index=False))
  print(f"\nAll outputs in {out_dir}")


def main(argv=None):
  args = parse_args(argv)
  out_dir = args.out_dir or os.path.join(args.vtr_dir, "hallucination")
  os.makedirs(out_dir, exist_ok=True)
  if args.stage in ("all", "run"):
    if not args.coco_dir:
      raise SystemExit("--coco_dir is needed to run")
    stage_run(args, out_dir)
  if args.stage in ("all", "summarize"):
    stage_summarize(args, out_dir)


if __name__ == "__main__":
  main()
