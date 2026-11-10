r"""Causal check (Task 7): does the model use an object's visual tokens at the layers where the
readouts can decode the object?

Per benchmark image (its one target object):
  clean     the image; P(class) for "What is the main object in this image? Answer with one word."
  masked    the object's pixels (mask dilated by --dilate px) painted the CLIP mean grey
  restore   the masked image, but at ONE decoder layer the object's tokens (every token the dilated
            mask touches) get their clean states back; one run per layer
  recovery  (p_restore - p_masked) / (p_clean - p_masked) for the true class, per layer

Restore modes (which image tokens get their clean states back at the one layer):
  object      every token the dilated mask touches
  all_image   all 576 image tokens (an upper bound: past the layer where the text positions have
              read the image, even that cannot bring the answer back)
  context     every token the dilated mask does NOT touch: is the object's information, by then,
              in other tokens (all_image recovers more than object mid-stack)?
  decoded / undecoded / random_k   at --token_layers only, among the object's tokens with
              coverage >= 0.5: k tokens the probe decodes (top-1 = the object), k it does not,
              and k at random (k = the smaller of the two counts). If decodability points at the
              tokens the model relies on, decoded recovers more than undecoded at the same k.

Two read-only experiments on the same images (--experiments), at --read_layers, read by a linear
probe (StandardScaler + logistic regression on the benchmark's cached object tokens, the chosen
images left out) and by the logit lens (final norm + LM head, the fair classes' first
sub-tokens):
  masked_read  read the MASKED image's tokens: at the object's positions (now grey) and at the
               tokens the mask does not touch. If they still name the object, the readout reports
               context (or information that flowed in from elsewhere), not the object's pixels.
               The clean image's same tokens are the reference.
  paste        paste the object's pixels into another benchmark image (a different class) at the
               same place: do the pasted object's tokens name the object, or the new scene's
               object? Also the model's own answer to the main-object question.

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
    -> <vtr_dir>/causal_restore/{images.csv, restores.csv, curve.csv, cases.csv, cases_by_layer.csv,
                                 tokens.csv, reads.csv, paste.csv, masked_read.csv, paste_read.csv}

Resumable: images already in restores.csv are skipped. --stage summarize only re-reads the CSVs.
"""

import argparse
import json
import os
import pickle
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
from vtr.geometry import HIGH_COVERAGE, Qwen25Geometry  # noqa: E402
from vtr.injection import IMAGE_QUESTION  # noqa: E402
from vtr.models import chat_inputs  # noqa: E402
from vtr.oracle import GREY, class_scores, first_tokens, square_crop  # noqa: E402
from vtr.readouts import class_forms  # noqa: E402

READOUTS = ["logit_lens", "embedding_lens", "latentlens", "patchscopes_pmi", "selfie_pmi",
            "image_slot_pmi", "oracle_pmi", "probe_linear"]
CASES = ["A", "B", "C", "D"]
TOKEN_MODES = ("decoded", "undecoded", "random_k")


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
  p.add_argument("--token_layers", default="0,2,4,6,8,10,12,14,16",
                 help="layers of the decoded / undecoded / random_k restores")
  p.add_argument("--experiments", default="",
                 help="comma-separated: masked_read, paste (read-only, see the docstring)")
  p.add_argument("--read_layers", default="0,4,8,12,16,20,24,28,31")
  p.add_argument("--dilate", type=int, default=4, help="mask dilation in original pixels")
  p.add_argument("--batch", type=int, default=8, help="restore runs per forward pass")
  p.add_argument("--min_gap", type=float, default=0.2, help="p_clean - p_masked for a valid image")
  p.add_argument("--used_threshold", type=float, default=0.5)
  p.add_argument("--readouts", default=",".join(READOUTS))
  p.add_argument("--min_pixels", type=int, default=56 * 56, help="Qwen2.5-VL resize lower bound")
  p.add_argument("--max_pixels", type=int, default=12845056,
                 help="Qwen2.5-VL resize upper bound; use the value of the benchmark run "
                      "(job_vtr_qwen.sh: 512 * 784)")
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  return p.parse_args(argv)


# ---------------------------------------------------------------------------------- images


class TokenGrid:
  """How an image becomes the model's input image and which tokens a mask touches.

  LLaVA-1.5: the 336 px centre crop (square_crop) and its 24x24 grid (patch_coverage).
  Qwen2.5-VL: the whole image (the processor resizes it; no crop) and its rows x cols grid of
  28 px cells (Qwen25Geometry), so masks are measured on the original image."""

  def __init__(self, family, min_pixels=None, max_pixels=None):
    self.qwen = family == "qwen2.5-vl"
    bounds = {k: v for k, v in (("min_pixels", min_pixels), ("max_pixels", max_pixels))
              if v is not None}
    self.geometry = Qwen25Geometry(**bounds) if self.qwen else None

  def model_image(self, image):
    return image if self.qwen else square_crop(image)

  def coverage(self, mask):
    """Per-token coverage of a mask in original image coordinates, flattened."""
    if self.qwen:
      return self.geometry.coverage(mask)
    return patch_coverage(mask).reshape(-1)


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
  """First-answer-step class log-probs for the main-object question on one image, n copies.
  LLaVA-1.5 or Qwen2.5-VL (vtr.models.chat_inputs builds either chat prompt)."""

  def __init__(self, mt, kept):
    self.mt, self.kept = mt, kept
    self.cls, self.ids = first_tokens(mt.tokenizer, kept, class_forms(kept, synonyms=False))
    self.layers = get_decoder(mt).layers
    self.n_tokens = mt.num_visual_tokens  # Qwen: set per image by inputs()

  def inputs(self, image, n):
    x, start, n_tokens = chat_inputs(self.mt, [image] * n, IMAGE_QUESTION)
    if self.mt.num_visual_tokens and n_tokens != self.mt.num_visual_tokens:
      raise ValueError(f"expected {self.mt.num_visual_tokens} image tokens, found {n_tokens}")
    self.n_tokens = n_tokens
    return x, start

  @torch.no_grad()
  def logp(self, x):
    out = self.mt.model(**x, use_cache=False)
    logp = out.logits[:, -1].float().log_softmax(-1).cpu()
    return class_scores(logp, self.cls, self.ids, len(self.kept))  # [n, n_kept]

  def clean(self, image):
    x, start = self.inputs(image, 1)
    store = {}
    with capture_image_states(self.layers, start, self.n_tokens, store):
      scores = self.logp(x)
    return scores[0], torch.stack([store[l] for l in range(len(self.layers))])

  def masked(self, image):
    x, _ = self.inputs(image, 1)
    return self.logp(x)[0]

  @torch.no_grad()
  def lens(self, h):
    """Logit lens on states [n, d] -> class scores [n, n_kept]."""
    dec = get_decoder(self.mt)
    logits = self.mt.model.get_output_embeddings()(dec.norm(h.to(self.mt.device)))
    return class_scores(logits.float().log_softmax(-1).cpu(), self.cls, self.ids, len(self.kept))

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


def parse_layers(spec, n_layers):
  return list(range(n_layers)) if spec == "all" else [int(x) for x in spec.split(",") if x]


def fit_probes(vtr_dir, out_dir, kept, exclude, layers):
  """{layer: probe}: StandardScaler + logistic regression on the benchmark's cached object tokens
  (coverage >= 0.5, fair classes; labels index `kept`), images in `exclude` left out. Cached in
  <out_dir>/probes.pkl; refitted if a chosen image was not left out."""
  from sklearn.linear_model import LogisticRegression
  from sklearn.pipeline import make_pipeline
  from sklearn.preprocessing import StandardScaler
  path = os.path.join(out_dir, "probes.pkl")
  saved = {"exclude": set(), "probes": {}}
  if os.path.exists(path):
    with open(path, "rb") as f:
      saved = pickle.load(f)
    if not set(exclude) <= saved["exclude"]:
      saved = {"exclude": set(), "probes": {}}
  todo = [l for l in layers if l not in saved["probes"]]
  if not todo:
    return saved["probes"]
  cache = torch.load(os.path.join(vtr_dir, "states.pt"), weights_only=False)
  meta = cache["meta"]
  rows = np.flatnonzero(((meta["kind"] == "object") & (meta["coverage"] >= HIGH_COVERAGE)
                         & meta["class_name"].isin(kept)
                         & ~meta["image_id"].isin(set(exclude))).to_numpy())
  y = meta["class_name"].iloc[rows].map({c: i for i, c in enumerate(kept)}).to_numpy()
  for l in todo:
    X = cache["states"][rows, l].float().numpy()
    saved["probes"][l] = make_pipeline(StandardScaler(),
                                       LogisticRegression(max_iter=1000)).fit(X, y)
    print(f"  probe layer {l}: {len(rows)} object tokens of "
          f"{meta['image_id'].iloc[rows].nunique()} images", flush=True)
  saved["exclude"] = set(exclude)
  with open(path, "wb") as f:
    pickle.dump(saved, f)
  return saved["probes"]


def probe_predict(probe, h):
  """h [n, d] -> predicted class index (into kept) [n]."""
  return probe.predict(h.float().cpu().numpy())


def square_mask(mask, size=336):
  """The object mask under square_crop's geometry (nearest-neighbour), [size, size] bool."""
  h, w = mask.shape
  scale = size / min(w, h)
  nw, nh = int(round(w * scale)), int(round(h * scale))
  m = Image.fromarray(mask.astype(np.uint8) * 255).resize((nw, nh), Image.NEAREST)
  left, top = (nw - size) // 2, (nh - size) // 2
  return np.array(m.crop((left, top, left + size, top + size))) > 0


def grid_coverage(sq_mask, patch=14):
  """Per-token coverage [576] of a square 336 mask."""
  g = sq_mask.shape[0] // patch
  return sq_mask.reshape(g, patch, g, patch).mean((1, 3)).reshape(-1)


def token_restores(probes, clean_states, cand, y, layers, rng):
  """[(mode, layer, token indices, extra)] for decoded / undecoded / random_k at `layers`."""
  out = []
  for l in layers:
    pred = probe_predict(probes[l], clean_states[l, torch.as_tensor(cand)])
    dec, und = cand[pred == y], cand[pred != y]
    k = min(len(dec), len(und))
    if k == 0:
      continue
    for mode, pool in (("decoded", dec), ("undecoded", und), ("random_k", cand)):
      out.append((mode, l, np.sort(rng.choice(pool, k, replace=False)),
                  {"n_decoded": len(dec), "n_candidates": len(cand)}))
  return out


def read_rows(runner, probes, states, regions, y, layers, scene=None, **key):
  """Rows of reads.csv: per layer, region and reader, the share of the region's tokens whose
  top-1 is the object (target) and, if `scene` is given, the scene's class."""
  rows = []
  for l in layers:
    for region, pos in regions.items():
      if not len(pos):
        continue
      h = states[l, torch.as_tensor(pos)]
      for reader, pred in (("probe", probe_predict(probes[l], h)),
                           ("logit_lens", runner.lens(h).argmax(1))):
        rows.append({**key, "layer": l, "region": region, "reader": reader, "n": len(pos),
                     "target": float(np.mean(pred == y)),
                     "scene": float(np.mean(pred == scene)) if scene is not None else np.nan})
  return rows


def append(path, rows):
  if rows:
    pd.DataFrame(rows).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def stage_run(args, out_dir, data, kept):
  from train_oracle import load_model
  modes = [m for m in args.modes.split(",") if m]
  experiments = [e for e in args.experiments.split(",") if e]
  res_path = os.path.join(out_dir, "restores.csv")
  if os.path.exists(res_path) and "n_tokens" not in pd.read_csv(res_path, nrows=1).columns:
    raise SystemExit(f"{res_path} is from an older version of this script: use another --out_dir")
  chosen = choose_images(data["samples"], kept, args.n_images, args.seed)
  mt = load_model(args)
  mt.model.eval()
  if getattr(mt, "family", "llava") == "qwen2.5-vl":
    mt.processor.set_pixel_bounds(args.min_pixels, args.max_pixels)
    if experiments:
      raise SystemExit("masked_read / paste use LLaVA's 336 px crop: run them on LLaVA only")
  grid = TokenGrid(getattr(mt, "family", "llava"), args.min_pixels, args.max_pixels)
  runner = Runner(mt, kept)
  n_layers = len(runner.layers)
  layers = parse_layers(args.layers, n_layers)
  token_layers = parse_layers(args.token_layers, n_layers) if set(modes) & set(TOKEN_MODES) else []
  read_layers = parse_layers(args.read_layers, n_layers) if experiments else []
  probes = (fit_probes(args.vtr_dir, out_dir, kept, [s["image_id"] for s in chosen],
                       sorted(set(token_layers) | set(read_layers)))
            if token_layers or read_layers else {})
  with open(ensure_annotations(args.coco_dir), encoding="utf-8") as f:
    anns = {a["id"]: a for a in json.load(f)["annotations"]}

  img_path = os.path.join(out_dir, "images.csv")
  reads_path, paste_path = os.path.join(out_dir, "reads.csv"), os.path.join(out_dir, "paste.csv")
  done = set(pd.read_csv(img_path)["image_id"]) if os.path.exists(img_path) else set()
  print(f"  {len(chosen)} images, {len(layers)} layers, modes {modes}, experiments "
        f"{experiments}; {len(done)} already done")
  for k, s in enumerate(chosen):
    if s["image_id"] in done:
      continue
    t0 = time.time()
    image = Image.open(ensure_image(args.coco_dir, s["file_name"])).convert("RGB")
    mask = ann_to_mask(anns[s["ann_id"]], image.height, image.width)
    masked_full, grown = mask_object(image, mask, args.dilate)
    tokens = np.flatnonzero(grid.coverage(grown) > 0)
    cov = grid.coverage(mask)
    cand = np.flatnonzero(cov >= HIGH_COVERAGE)
    clean_img, masked_img = grid.model_image(image), grid.model_image(masked_full)
    y = kept.index(s["class_name"])
    rng = np.random.RandomState(args.seed + int(s["image_id"]))

    clean_scores, clean_states = runner.clean(clean_img)
    masked_scores, masked_states = runner.clean(masked_img)
    if clean_states.shape[1] != len(cov):
      raise ValueError(f"image {s['image_id']}: {clean_states.shape[1]} image tokens but "
                       f"{len(cov)} grid cells (check --min_pixels / --max_pixels)")
    every = np.arange(len(cov))
    base = {"object": tokens, "all_image": every, "context": np.setdiff1d(every, tokens)}
    specs = [(m, l, base[m], {}) for m in modes if m in base for l in layers]
    if token_layers:
      specs += token_restores(probes, clean_states, cand, y, token_layers, rng)
    jobs = [(l, torch.as_tensor(tok, device=mt.device)) for _, l, tok, _ in specs]
    rest = runner.restored(masked_img, clean_states, jobs, args.batch) if jobs else []

    p_c, p_m = np.exp(clean_scores[y]), np.exp(masked_scores[y])
    append(res_path, [
        {"image_id": s["image_id"], "mode": m, "layer": l, "p": float(np.exp(sc[y])),
         "top1": bool(sc.argmax() == y),
         "recovery": float((np.exp(sc[y]) - p_m) / (p_c - p_m)) if p_c != p_m else np.nan,
         "n_tokens": len(tok), "mean_cov": float(cov[tok].mean()),
         "n_decoded": extra.get("n_decoded", np.nan),
         "n_candidates": extra.get("n_candidates", np.nan)}
        for (m, l, tok, extra), sc in zip(specs, rest)])

    reads, pasted = [], []
    if "masked_read" in experiments:
      regions = {"object": cand,
                 "outside": np.flatnonzero(patch_coverage(grown).reshape(-1) == 0)}
      for name, st in (("clean", clean_states), ("masked", masked_states)):
        reads += read_rows(runner, probes, st, regions, y, read_layers,
                           experiment="masked_read", image_id=s["image_id"], image=name)
    if "paste" in experiments:
      donor = next(d for d in chosen[k + 1:] + chosen[:k] if d["class_name"] != s["class_name"])
      scene = Image.open(ensure_image(args.coco_dir, donor["file_name"])).convert("RGB")
      sq = square_mask(mask)
      composite = square_crop(scene)
      composite.paste(clean_img, mask=Image.fromarray(sq.astype(np.uint8) * 255))
      c_scores, c_states = runner.clean(composite)
      yb = kept.index(donor["class_name"])
      gcov = grid_coverage(sq)
      regions = {"object": np.flatnonzero(gcov >= HIGH_COVERAGE),
                 "scene": np.flatnonzero(gcov == 0)}
      reads += read_rows(runner, probes, c_states, regions, y, read_layers, scene=yb,
                         experiment="paste", image_id=s["image_id"], image="pasted")
      pasted.append({"image_id": s["image_id"], "donor_id": donor["image_id"],
                     "class_name": s["class_name"], "donor_class": donor["class_name"],
                     "p_object": float(np.exp(c_scores[y])),
                     "p_scene": float(np.exp(c_scores[yb])),
                     "answer": kept[int(c_scores.argmax())]})
    append(reads_path, reads)
    append(paste_path, pasted)
    append(img_path, [{"image_id": s["image_id"], "class_name": s["class_name"],
                       "n_restored": len(tokens), "p_clean": float(p_c), "p_masked": float(p_m),
                       "top1_clean": bool(clean_scores.argmax() == y),
                       "top1_masked": bool(masked_scores.argmax() == y),
                       "masked_answer": kept[int(masked_scores.argmax())]}])
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


def bootstrap_ci(x, seed, n_boot=2000):
  x = np.asarray(x, dtype=float)
  if len(x) < 2:
    return np.nan, np.nan
  rng = np.random.RandomState(seed)
  means = x[rng.randint(0, len(x), (n_boot, len(x)))].mean(1)
  return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize_tokens(out_dir, restores, seed):
  """decoded vs undecoded vs random_k restores at the same k, per layer (paired by image)."""
  t = restores[restores["mode"].isin(TOKEN_MODES)]
  if t.empty:
    return
  rows = []
  for l, g in t.groupby("layer"):
    w = g.pivot_table(index="image_id", columns="mode", values="recovery").dropna()
    if w.empty:
      continue
    d = w["decoded"] - w["undecoded"]
    lo, hi = bootstrap_ci(d, seed)
    cov = g.groupby("mode")["mean_cov"].mean()
    rows.append({"layer": l, "n_images": len(w), "k_mean": float(g["n_tokens"].mean()),
                 **{m: float(w[m].mean()) for m in TOKEN_MODES},
                 "decoded_minus_undecoded": float(d.mean()), "ci_lo": lo, "ci_hi": hi,
                 **{f"cov_{m}": float(cov.get(m, np.nan)) for m in TOKEN_MODES}})
  tab = pd.DataFrame(rows).set_index("layer")
  tab.to_csv(os.path.join(out_dir, "tokens.csv"))
  print("\nRestore only k of the object's tokens (coverage >= 0.5): the ones the probe decodes,\n"
        "the ones it does not, or k at random; recovery, paired over images with both kinds:")
  print(tab.round(2).to_string())


def summarize_reads(out_dir, valid_ids, clean_ids):
  """Masked-image and pasted-object reads: share of tokens whose top-1 is the object."""
  path = os.path.join(out_dir, "reads.csv")
  if not os.path.exists(path):
    return
  reads = pd.read_csv(path)
  m = reads[(reads["experiment"] == "masked_read") & reads["image_id"].isin(valid_ids)]
  if not m.empty:
    t = m.groupby(["reader", "region", "image", "layer"])["target"].mean().unstack("layer")
    t.to_csv(os.path.join(out_dir, "masked_read.csv"))
    print("\nMasked image: share of tokens read as the object (probe / logit lens), at the\n"
          "object's positions (grey after masking) and at tokens the mask does not touch:")
    print(t.round(2).to_string())
  p = reads[(reads["experiment"] == "paste") & reads["image_id"].isin(clean_ids)]
  if not p.empty:
    t = (p.groupby(["reader", "region", "layer"])[["target", "scene"]].mean()
         .unstack("layer"))
    t.to_csv(os.path.join(out_dir, "paste_read.csv"))
    print("\nObject pasted into another image: share of tokens read as the pasted object\n"
          "(target) and as the new scene's object (scene):")
    print(t.round(2).to_string())
    ans = pd.read_csv(os.path.join(out_dir, "paste.csv"))
    ans = ans[ans["image_id"].isin(clean_ids)]
    print(f"  model's main-object answer on the composite ({len(ans)} images): pasted object "
          f"{(ans['answer'] == ans['class_name']).mean():.2f}, scene's object "
          f"{(ans['answer'] == ans['donor_class']).mean():.2f}")


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

  summarize_tokens(out_dir, restores, args.seed)
  summarize_reads(out_dir, set(valid["image_id"]), set(images.loc[images["top1_clean"], "image_id"]))

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
