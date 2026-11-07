r"""Causal check (Task 7): does the model use an object's visual tokens at the layers where the
readouts can decode the object?

Per benchmark image (its one target object):
  clean     the image; P(class) for "What is the main object in this image? Answer with one word."
  masked    the object's pixels (mask dilated by --dilate px) painted the CLIP mean grey
  restore   the masked image, but at ONE decoder layer the object's tokens (every token the dilated
            mask touches) get their clean states back; one run per layer
  recovery  (p_restore - p_masked) / (p_clean - p_masked) for the true class, per layer

Mode "all_image" restores all 576 image tokens instead (an upper bound: past the layer where the
text positions have read the image, even that cannot bring the answer back).

P(class) is the probability of the class's best first sub-token at the first answer step, over the
fair classes, as in train_oracle.py's main-object question. Images count as valid when the clean
image is answered correctly (closed-set top-1) and masking costs at least --min_gap of P(class).

The comparison with readouts uses the saved benchmark scores (<vtr_dir>/scores/<name>.npz): a
readout "decodes" the object at a layer when it ranks the true class first (fair classes) on at
least half of the object's cached tokens (coverage >= 0.5). The model "uses" the object tokens at
that layer when recovery >= --used_threshold. Each (image, layer) is then one of
  A decodes and used      B decodes, not used      C used, not decoded      D neither
and each readout gets two hit rates: P(decodes | used) and P(decodes | not used). A readout that
tracks what the model uses has the first high and the second low.

  python causal_restore.py --coco_dir /path/to/coco --vtr_dir ./results/vtr_llava15 --n_images 50
    -> <vtr_dir>/causal_restore/{images.csv, restores.csv, curve.csv, cases.csv, cases_by_layer.csv}

Resumable: images already in restores.csv are skipped. --stage summarize only re-reads the CSVs.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_vtr  # noqa: E402
from train_oracle import load_benchmark  # noqa: E402
from visual_object_identification import (ann_to_mask, ensure_annotations, ensure_image,  # noqa: E402
                                          get_decoder, patch_coverage)
from vtr import evaluate as E  # noqa: E402
from vtr.geometry import HIGH_COVERAGE  # noqa: E402
from vtr.injection import IMAGE_QUESTION  # noqa: E402
from vtr.oracle import GREY, class_scores, first_tokens, square_crop  # noqa: E402
from vtr.readouts import class_forms  # noqa: E402

READOUTS = ["logit_lens", "embedding_lens", "latentlens", "patchscopes_pmi", "selfie_pmi",
            "image_slot_pmi", "oracle_pmi", "probe_linear"]
CASES = ["A", "B", "C", "D"]


def parse_args(argv=None):
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--stage", default="all", choices=["all", "run", "summarize"])
  p.add_argument("--coco_dir", default=None, help="needed for --stage run/all")
  p.add_argument("--vtr_dir", default="./results/vtr_llava15")
  p.add_argument("--out_dir", default=None, help="default <vtr_dir>/causal_restore")
  p.add_argument("--model_name", default="llava-hf/llava-1.5-7b-hf")
  p.add_argument("--n_images", type=int, default=50)
  p.add_argument("--layers", default="all", help="'all' or comma-separated restore layers")
  p.add_argument("--modes", default="object,all_image")
  p.add_argument("--dilate", type=int, default=4, help="mask dilation in original pixels")
  p.add_argument("--batch", type=int, default=8, help="restore runs per forward pass")
  p.add_argument("--min_gap", type=float, default=0.2, help="p_clean - p_masked for a valid image")
  p.add_argument("--used_threshold", type=float, default=0.5)
  p.add_argument("--readouts", default=",".join(READOUTS))
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  return p.parse_args(argv)


# ---------------------------------------------------------------------------------- images


def mask_object(image, mask, dilate):
  """(masked image, dilated mask): mask pixels (grown by `dilate` px) painted CLIP mean grey."""
  m = Image.fromarray(mask.astype(np.uint8) * 255)
  if dilate > 0:
    m = m.filter(ImageFilter.MaxFilter(2 * dilate + 1))
  out = image.copy()
  out.paste(Image.new("RGB", image.size, GREY), mask=m)
  return out, np.array(m) > 0


def restore_positions(mask):
  """Image-token indices (0..575) whose patch the mask touches at all."""
  return np.flatnonzero(patch_coverage(mask).reshape(-1) > 0)


# ---------------------------------------------------------------------------------- model


@contextmanager
def restore_states(layers, start, plan):
  """plan[row] = (layer, token indices [k], states [k, d]) or None. Overwrites the output of
  decoder block `layer` at sequence positions start + tokens, for that batch row only."""
  by_layer = defaultdict(list)
  for row, item in enumerate(plan):
    if item is not None:
      by_layer[item[0]].append((row, item[1], item[2]))

  def make(entries):
    def hook(module, inp, out):
      hs = out[0] if isinstance(out, tuple) else out
      hs = hs.clone()
      for row, tokens, st in entries:
        hs[row, start + tokens.to(hs.device)] = st.to(device=hs.device, dtype=hs.dtype)
      return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs
    return hook

  handles = [layers[l].register_forward_hook(make(e)) for l, e in by_layer.items()]
  try:
    yield
  finally:
    for h in handles:
      h.remove()


@contextmanager
def capture_image_states(layers, start, n_tokens, store):
  """store[l] = output of decoder block l at the image positions of batch row 0, [n_tokens, d]."""
  def make(l):
    def hook(module, inp, out):
      hs = out[0] if isinstance(out, tuple) else out
      store[l] = hs[0, start:start + n_tokens].detach().clone()
    return hook

  handles = [layer.register_forward_hook(make(l)) for l, layer in enumerate(layers)]
  try:
    yield
  finally:
    for h in handles:
      h.remove()


class Runner:
  """First-answer-step class log-probs for the main-object question on one image, n copies."""

  def __init__(self, mt, kept):
    self.mt, self.kept = mt, kept
    self.cls, self.ids = first_tokens(mt.tokenizer, kept, class_forms(kept, synonyms=False))
    self.prompt = f"USER: <image>\n{IMAGE_QUESTION} ASSISTANT:"
    self.layers = get_decoder(mt).layers

  def inputs(self, image, n):
    x = self.mt.processor(images=[image] * n, text=[self.prompt] * n, return_tensors="pt")
    x = {k: v.to(self.mt.device) for k, v in x.items()}
    x["pixel_values"] = x["pixel_values"].to(self.mt.model.dtype)
    ids = x["input_ids"][0]
    pos = (ids == self.mt.model.config.image_token_index).nonzero(as_tuple=True)[0]
    if len(pos) != self.mt.num_visual_tokens:
      raise ValueError(f"expected {self.mt.num_visual_tokens} image tokens, found {len(pos)}")
    return x, int(pos[0])

  @torch.no_grad()
  def logp(self, x):
    out = self.mt.model(**x, use_cache=False)
    logp = out.logits[:, -1].float().log_softmax(-1).cpu()
    return class_scores(logp, self.cls, self.ids, len(self.kept))  # [n, n_kept]

  def clean(self, image):
    x, start = self.inputs(image, 1)
    store = {}
    with capture_image_states(self.layers, start, self.mt.num_visual_tokens, store):
      scores = self.logp(x)
    return scores[0], torch.stack([store[l] for l in range(len(self.layers))])

  def masked(self, image):
    x, _ = self.inputs(image, 1)
    return self.logp(x)[0]

  def restored(self, image, clean_states, jobs, batch):
    """jobs: [(layer, token indices)] -> scores [len(jobs), n_kept]."""
    out = []
    for b in range(0, len(jobs), batch):
      chunk = jobs[b:b + batch]
      x, start = self.inputs(image, len(chunk))
      plan = [(l, tok, clean_states[l, tok]) for l, tok in chunk]
      with restore_states(self.layers, start, plan):
        out.append(self.logp(x))
    return np.concatenate(out)


# ---------------------------------------------------------------------------------- run


def choose_images(samples, kept, n, seed):
  pool = [s for s in samples if s["class_name"] in kept]
  rng = np.random.RandomState(seed)
  return [pool[i] for i in rng.permutation(len(pool))[:n]]


def stage_run(args, out_dir, data, kept):
  from train_oracle import load_model
  mt = load_model(args)
  mt.model.eval()
  runner = Runner(mt, kept)
  n_layers = len(runner.layers)
  layers = list(range(n_layers)) if args.layers == "all" else [int(x) for x in
                                                               args.layers.split(",")]
  modes = args.modes.split(",")
  with open(ensure_annotations(args.coco_dir), encoding="utf-8") as f:
    anns = {a["id"]: a for a in json.load(f)["annotations"]}

  img_path, res_path = os.path.join(out_dir, "images.csv"), os.path.join(out_dir, "restores.csv")
  done = set(pd.read_csv(img_path)["image_id"]) if os.path.exists(img_path) else set()
  chosen = choose_images(data["samples"], kept, args.n_images, args.seed)
  print(f"  {len(chosen)} images, {len(layers)} layers, modes {modes}; {len(done)} already done")
  for k, s in enumerate(chosen):
    if s["image_id"] in done:
      continue
    t0 = time.time()
    image = Image.open(ensure_image(args.coco_dir, s["file_name"])).convert("RGB")
    mask = ann_to_mask(anns[s["ann_id"]], image.height, image.width)
    masked_full, grown = mask_object(image, mask, args.dilate)
    tokens = torch.as_tensor(restore_positions(grown), device=mt.device)
    clean_img, masked_img = square_crop(image), square_crop(masked_full)
    y = kept.index(s["class_name"])

    clean_scores, clean_states = runner.clean(clean_img)
    masked_scores = runner.masked(masked_img)
    every = torch.arange(mt.num_visual_tokens, device=mt.device)
    jobs = [(l, tokens if m == "object" else every) for m in modes for l in layers]
    rest = runner.restored(masked_img, clean_states, jobs, args.batch)

    p_c, p_m = np.exp(clean_scores[y]), np.exp(masked_scores[y])
    rows = [{"image_id": s["image_id"], "mode": m, "layer": l, "p": float(np.exp(sc[y])),
             "top1": bool(sc.argmax() == y),
             "recovery": float((np.exp(sc[y]) - p_m) / (p_c - p_m)) if p_c != p_m else np.nan}
            for (m, l), sc in zip([(m, l) for m in modes for l in layers], rest)]
    pd.DataFrame(rows).to_csv(res_path, mode="a", header=not os.path.exists(res_path),
                              index=False)
    pd.DataFrame([{"image_id": s["image_id"], "class_name": s["class_name"],
                   "n_restored": len(tokens), "p_clean": float(p_c), "p_masked": float(p_m),
                   "top1_clean": bool(clean_scores.argmax() == y),
                   "top1_masked": bool(masked_scores.argmax() == y),
                   "masked_answer": kept[int(masked_scores.argmax())]}]).to_csv(
        img_path, mode="a", header=not os.path.exists(img_path), index=False)
    print(f"  [{k + 1}/{len(chosen)}] {s['class_name']}: p clean {p_c:.2f} masked {p_m:.2f}, "
          f"{len(tokens)} tokens ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------------------------- summary


def readout_hits(vtr_dir, names, meta, class_names, kept, image_ids):
  """Long table (readout, image_id, layer, frac): share of the object's cached tokens
  (coverage >= 0.5) on which the readout ranks the true class first among the fair classes."""
  index = {c: i for i, c in enumerate(class_names)}
  keep = [index[c] for c in kept]
  rows = np.flatnonzero((meta["image_id"].isin(image_ids) & (meta["kind"] == "object")
                         & (meta["coverage"] >= HIGH_COVERAGE)
                         & meta["class_name"].isin(kept)).to_numpy())
  lab = meta["class_name"].iloc[rows].map({c: i for i, c in enumerate(kept)}).to_numpy()
  img = meta["image_id"].iloc[rows].to_numpy()
  out = []
  for name in names:
    path = os.path.join(vtr_dir, "scores", f"{name}.npz")
    if not os.path.exists(path):
      print(f"  (no {path}: {name} skipped)")
      continue
    layers, scores = run_vtr.load_scores(path)
    for li, l in enumerate(layers):
      hit = E.ranks(scores[li][rows][:, keep], lab) == 1
      frac = pd.Series(hit).groupby(img).mean()
      out += [{"readout": name, "image_id": i, "layer": int(l), "frac": float(f)}
              for i, f in frac.items()]
  return pd.DataFrame(out)


def case_tables(restores, hits, used_threshold):
  """(per readout, per readout and layer) on the object-mode restores: the A/B/C/D shares and
  the two hit rates, P(decodes | used) and P(decodes | not used). (A + D alone would reward a
  readout that never decodes, since late layers are almost all "not used".)"""
  r = restores[restores["mode"] == "object"][["image_id", "layer", "recovery"]]
  m = hits.merge(r, on=["image_id", "layer"])
  m["dec"], m["used"] = m["frac"] >= 0.5, m["recovery"] >= used_threshold
  m["case"] = np.select([m.dec & m.used, m.dec & ~m.used, ~m.dec & m.used], ["A", "B", "C"], "D")

  def table(keys):
    g = m.groupby(keys)
    t = g["case"].value_counts(normalize=True).unstack(fill_value=0.0)
    t = t.reindex(columns=CASES, fill_value=0.0)
    t["hit_used"] = m[m.used].groupby(keys)["dec"].mean()
    t["hit_unused"] = m[~m.used].groupby(keys)["dec"].mean()
    t["n_used"], t["n"] = m[m.used].groupby(keys).size(), g.size()
    return t

  return table(["readout"]), table(["readout", "layer"])


def stage_summarize(args, out_dir, data, kept):
  images = pd.read_csv(os.path.join(out_dir, "images.csv"))
  restores = pd.read_csv(os.path.join(out_dir, "restores.csv"))
  valid = images[images["top1_clean"] & (images["p_clean"] - images["p_masked"] >= args.min_gap)]
  print(f"\n{len(valid)} of {len(images)} images valid (clean top-1 correct, masking costs >= "
        f"{args.min_gap} of P(class)); masked top-1 still correct on "
        f"{int(images['top1_masked'].sum())}")
  restores = restores[restores["image_id"].isin(valid["image_id"])]
  curve = restores.groupby(["mode", "layer"]).agg(recovery_mean=("recovery", "mean"),
                                                 recovery_median=("recovery", "median"),
                                                 top1=("top1", "mean")).reset_index()
  curve.to_csv(os.path.join(out_dir, "curve.csv"), index=False)
  pd.set_option("display.width", 200)
  print("\nRecovery by restore layer (mean over valid images):")
  print(curve.pivot(index="layer", columns="mode", values="recovery_mean").round(2).T.to_string())

  cache = torch.load(os.path.join(args.vtr_dir, "states.pt"), weights_only=False)
  hits = readout_hits(args.vtr_dir, args.readouts.split(","), cache["meta"],
                      data["class_names"], kept, set(valid["image_id"]))
  if hits.empty:
    print("no readout scores found")
    return
  cases, by_layer = case_tables(restores, hits, args.used_threshold)
  cases.to_csv(os.path.join(out_dir, "cases.csv"))
  by_layer.to_csv(os.path.join(out_dir, "cases_by_layer.csv"))
  print(f"\nOver (image, layer); used = recovery >= {args.used_threshold}, decodes = top-1 on >= "
        "half the object's tokens:")
  print("  A decodes & used   B decodes, not used   C used, not decoded   D neither")
  print("  hit_used = P(decodes | used) (low: misses what the model uses)")
  print("  hit_unused = P(decodes | not used) (high: reports what the model no longer uses)")
  print(cases.sort_values("hit_used", ascending=False).round(2).to_string())
  used_rate = by_layer.groupby("layer")["n_used"].first() / by_layer.groupby("layer")["n"].first()
  decode = (by_layer["A"] + by_layer["B"]).unstack("layer")
  decode.loc["(model uses: share of images)"] = used_rate.fillna(0.0)
  print("\nDecode rate by layer, next to how often the model uses the object tokens:")
  print(decode.round(2).to_string())


def main(argv=None):
  args = parse_args(argv)
  out_dir = args.out_dir or os.path.join(args.vtr_dir, "causal_restore")
  os.makedirs(out_dir, exist_ok=True)
  data, kept = load_benchmark(args.vtr_dir)
  if args.stage in ("all", "run"):
    if not args.coco_dir:
      raise SystemExit("--coco_dir is needed to run")
    stage_run(args, out_dir, data, kept)
  if args.stage in ("all", "summarize"):
    stage_summarize(args, out_dir, data, kept)
  print(f"\nAll outputs in {out_dir}")


if __name__ == "__main__":
  main()
