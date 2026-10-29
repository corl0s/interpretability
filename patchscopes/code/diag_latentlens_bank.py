r"""Why does LatentLens score lower with one bank than another on COCO-VTR? Four checks, on the
object_high rows of the fair classes, from the cached states (no LLaVA forward passes):

  1. coverage  bank entries per class in each bank
  2. matched   the reference bank subsampled to the other bank's per-class counts: if accuracy
               falls to the other bank's level, coverage explains the gap; if not, the contexts
               themselves are worse
  3. layers    accuracy with one bank layer at a time (the readout takes the max over all)
  4. bias      share of rows whose top-1 is the most-predicted classes, and how that tracks the
               class's entry count

  python diag_latentlens_bank.py --out_dir ./results/vtr_llava15 \
      --ref ./results/latentlens_index/bank:latentlens \
      --new ./results/ll_llava15/bank:latentlens_vg
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vtr.evaluate import condition_masks  # noqa: E402
from vtr.readouts import LatentLens, _scatter_max, class_forms  # noqa: E402


def top1(ro, h, keep, rows=None, layers=None):
  """Top-1 class index (into keep) per row, using bank rows `rows` and bank layers `layers`."""
  q = torch.nn.functional.normalize(h.float(), dim=-1).to(ro.device)
  best = None
  for l, emb in ro.emb.items():
    if layers is not None and l not in layers:
      continue
    e, c = (emb, ro.cls) if rows is None else (emb[rows], ro.cls[rows])
    s = _scatter_max(q.to(e.dtype) @ e.T, c, ro.n_classes)
    best = s if best is None else torch.maximum(best, s)
  return best.float()[:, keep].argmax(1).cpu().numpy()


def main(argv=None):
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--out_dir", default="./results/vtr_llava15")
  p.add_argument("--ref", default="./results/latentlens_index/bank:latentlens",
                 help="bank_dir:readout_name (name locates the cached entry match)")
  p.add_argument("--new", default="./results/ll_llava15/bank:latentlens_vg")
  p.add_argument("--layers", default="0,8,16,22,28")
  p.add_argument("--model_name", default="llava-hf/llava-1.5-7b-hf")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  args = p.parse_args(argv)

  with open(os.path.join(args.out_dir, "samples.json"), encoding="utf-8") as f:
    class_names = json.load(f)["class_names"]
  with open(os.path.join(args.out_dir, "fair_classes.json"), encoding="utf-8") as f:
    kept = json.load(f)["kept"]
  keep = [class_names.index(c) for c in kept]
  cache = torch.load(os.path.join(args.out_dir, "states.pt"), weights_only=False)
  states, meta = cache["states"], cache["meta"]
  rows = condition_masks(meta)["object_high"] & meta["class_name"].isin(kept).to_numpy()
  labels = np.array([kept.index(c) for c in meta["class_name"][rows]])
  layers = [int(x) for x in args.layers.split(",")]
  print(f"{rows.sum()} object_high rows, {len(kept)} classes, layers {layers}")

  from transformers import AutoTokenizer
  tok = AutoTokenizer.from_pretrained(args.model_name)
  forms = class_forms(class_names, synonyms=False)
  banks = {}
  for spec in (args.ref, args.new):
    bank_dir, name = spec.rsplit(":", 1)
    banks[name] = LatentLens(bank_dir, tok, class_names, forms, args.device,
                             cache_file=os.path.join(args.out_dir, "scores",
                                                     f"{name}_entries.npz"))
  (ref_name, ref), (new_name, new) = banks.items()
  h = {l: states[rows, l] for l in layers}

  def acc(ro, **kw):
    return [100 * (top1(ro, h[l], keep, **kw) == labels).mean() for l in layers]

  def show(label, vals):
    print(f"  {label:34s}" + " ".join(f"{v:5.1f}" for v in vals) + f"   mean {np.mean(vals):5.1f}")

  # 1. coverage
  counts = {n: np.bincount(ro.cls.cpu().numpy(), minlength=len(class_names))[keep]
            for n, ro in banks.items()}
  print("\n1. Bank entries per class")
  for n, c in counts.items():
    print(f"  {n:14s} total {c.sum():8d}  min {c.min():6d}  median {int(np.median(c)):6d}  "
          f"max {c.max():6d}  zero {[k for k, x in zip(kept, c) if x == 0]}")
  ratio = counts[ref_name] / np.maximum(counts[new_name], 1)
  print("  fewest in", new_name, sorted(zip(counts[new_name], kept))[:8])
  print(f"  {ref_name}/{new_name} ratio per class: median {np.median(ratio):.1f}")

  # 2. matched counts
  print(f"\n2. Top-1 (%) by layer {layers}")
  show(f"{ref_name} (all entries)", acc(ref))
  show(f"{new_name} (all entries)", acc(new))
  rng = np.random.default_rng(args.seed)
  ref_cls = ref.cls.cpu().numpy()
  pick = []
  for j, ci in enumerate(keep):
    idx = np.flatnonzero(ref_cls == ci)
    pick += list(rng.choice(idx, min(len(idx), counts[new_name][j]), replace=False))
  pick = torch.as_tensor(np.sort(pick), device=ref.device)
  show(f"{ref_name} matched to {new_name} counts", acc(ref, rows=pick))
  for cap in (20, 100):
    pick = []
    for ci in keep:
      idx = np.flatnonzero(ref_cls == ci)
      pick += list(rng.choice(idx, min(len(idx), cap), replace=False))
    show(f"{ref_name} capped at {cap}/class", acc(ref, rows=torch.as_tensor(np.sort(pick),
                                                                           device=ref.device)))

  # 3. one bank layer at a time
  print("\n3. Top-1 (%) with a single bank layer")
  for n, ro in banks.items():
    for bl in ro.emb:
      show(f"{n} bank layer {bl}", acc(ro, layers=[bl]))

  # 4. prediction bias, at the middle query layer
  mid = layers[len(layers) // 2]
  print(f"\n4. Prediction bias at layer {mid}")
  for n, ro in banks.items():
    pred = top1(ro, h[mid], keep)
    share = np.bincount(pred, minlength=len(kept)) / len(pred)
    order = np.argsort(-share)[:5]
    x = np.log1p(counts[n])
    corr = np.corrcoef(x, share)[0, 1] if x.std() > 0 and share.std() > 0 else float("nan")
    print(f"  {n:14s} top-5 predicted: "
          + ", ".join(f"{kept[i]} {100 * share[i]:.0f}% ({counts[n][i]} entries)" for i in order)
          + f"; corr(log entries, share) {corr:+.2f}")

  # 5. which contexts: the class word's position in its sentence, and the sentence's length
  print("\n5. Matched entries by context (position of the class word; words in the sentence)")
  for spec, (n, ro) in zip((args.ref, args.new), banks.items()):
    meta_b = bank_metadata(spec.rsplit(":", 1)[0])
    cols = np.load(os.path.join(args.out_dir, "scores", f"{n}_entries.npz"))["cols"]
    pos = np.array([meta_b[j]["position"] for j in cols])
    words = np.array([len(meta_b[j]["caption"].split()) for j in cols])
    print(f"  {n}: position median {int(np.median(pos))}, sentence length median "
          f"{int(np.median(words))} words")
    for lo, hi in ((2, 4), (5, 9), (10, 10 ** 6)):
      sel = torch.as_tensor(np.flatnonzero((pos >= lo) & (pos <= hi)), device=ro.device)
      if len(sel):
        show(f"{n} position {lo}-{hi if hi < 10 ** 6 else ''} ({len(sel)})", acc(ro, rows=sel))
    dog = [j for j, c in zip(cols, ro.cls.cpu().numpy()) if class_names[c] == "dog"][:4]
    for j in dog:
      print(f"    dog @ {meta_b[j]['position']}: {meta_b[j]['caption'][:100]!r}")


def bank_metadata(bank_dir):
  """Entry metadata of a bank (identical for every layer; read from the first)."""
  import glob
  path = sorted(glob.glob(os.path.join(bank_dir, "layer_*", "embeddings_cache.pt"))
                + glob.glob(os.path.join(bank_dir, "*", "layer_*", "embeddings_cache.pt")))[0]
  return torch.load(path, map_location="cpu", weights_only=False, mmap=True)["metadata"]


if __name__ == "__main__":
  main()
