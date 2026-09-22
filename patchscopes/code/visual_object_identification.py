r"""Visual object identification: Patchscopes vs. logit lens vs. linear probe (LLaVA-1.5).

Step 1 of the evaluation plan in `Research/PROJECT_CONTEXT.md`.

Question: at each LLM backbone layer, can the object covered by a visual patch token be
read out of that token's hidden state? Three readouts are compared on the same states:

  1. Patchscopes  -- patch the state into a text-only target prompt, generate, and check
                     whether the COCO class name appears in the generation.
  2. Logit lens   -- project the state through the final norm + lm_head and check the
                     top-1 / top-5 vocabulary tokens against the class name.
  3. Linear probe -- supervised logistic regression on the same states (reference point).

Controls: patches outside the object, norm-matched random vectors, and the target prompt
generated with no injection at all (to expose what the prompt alone produces).

Protocol follows Neo et al. (ICLR 2025, arXiv 2410.07149 Sec. 4.1) -- COCO val2017 images,
one object of 20k-30k px^2 area, readout at visual tokens covering that object -- but it is
an approximation, not an exact replication: their matching criterion and image filter are
not fully specified in the paper. Target prompts are the unmodified ones from the original
Patchscopes notebooks (`next_token_prediction.ipynb`, `entity_processing.ipynb`).

Usage (SCC, one GPU, ~28GB is not needed -- 7B in fp16 is enough):

  python visual_object_identification.py --coco_dir /projectnb/cs505am/students/vishnuav/coco \
      --out_dir ./results/object_identification

  # quick smoke test first (3 images, 4 layers):
  python visual_object_identification.py --coco_dir ... --n_images 3 --layer_step 8 --debug_overlays

Stages are checkpointed; re-running skips finished stages unless --overwrite is passed.
"""

import argparse
import json
import os
import random
import re
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from general_utils import ModelAndTokenizer, make_inputs  # noqa: E402
from patchscopes_utils import (  # noqa: E402
    remove_hooks,
    set_hs_patch_hooks_llava_batch,
)

# The source text is irrelevant to the visual token states (causal attention: visual
# tokens cannot attend to text that follows them), but keep the official LLaVA-1.5 format.
SOURCE_PROMPT = "USER: <image>\nDescribe the image. ASSISTANT:"

# Unmodified target prompts from the original Patchscopes notebooks.
TARGET_PROMPTS = {
    "identity": "cat -> cat\n1135 -> 1135\nhello -> hello\n?",
    "entity": (
        "Syria: Country in the Middle East, Leonardo DiCaprio: American actor,"
        " Samsung: South Korean multinational major appliance and consumer"
        " electronics corporation, x"
    ),
}

COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMG_URL = "http://images.cocodataset.org/val2017/{file_name}"


# --------------------------------------------------------------------------------------
# COCO data preparation (no GPU needed)
# --------------------------------------------------------------------------------------


def ensure_annotations(coco_dir):
  """Return path to instances_val2017.json, downloading the annotation zip if needed."""
  ann_path = os.path.join(coco_dir, "annotations", "instances_val2017.json")
  if os.path.exists(ann_path):
    return ann_path
  os.makedirs(os.path.join(coco_dir, "annotations"), exist_ok=True)
  zip_path = os.path.join(coco_dir, "annotations_trainval2017.zip")
  if not os.path.exists(zip_path):
    print(f"Downloading COCO annotations (~241MB) to {zip_path} ...")
    urllib.request.urlretrieve(COCO_ANN_URL, zip_path)
  with zipfile.ZipFile(zip_path) as zf:
    zf.extract("annotations/instances_val2017.json", coco_dir)
  return ann_path


def ensure_image(coco_dir, file_name):
  """Return local path to a val2017 image, downloading the single file if needed."""
  img_dir = os.path.join(coco_dir, "val2017")
  os.makedirs(img_dir, exist_ok=True)
  path = os.path.join(img_dir, file_name)
  if not os.path.exists(path):
    urllib.request.urlretrieve(COCO_IMG_URL.format(file_name=file_name), path)
  return path


def ann_to_mask(ann, height, width):
  """Rasterize a non-crowd polygon annotation to a boolean mask (no pycocotools)."""
  if isinstance(ann["segmentation"], dict):  # RLE (crowd) -- not supported, filtered out
    return None
  mask_img = Image.new("L", (width, height), 0)
  draw = ImageDraw.Draw(mask_img)
  for poly in ann["segmentation"]:
    if len(poly) >= 6:
      draw.polygon(list(zip(poly[0::2], poly[1::2])), fill=1)
  return np.array(mask_img, dtype=bool)


def patch_coverage(mask, shortest_edge=336, crop=336, patch=14):
  """Fraction of each visual patch covered by the mask, as a (grid, grid) array.

  Mirrors the CLIPImageProcessor geometry of llava-hf/llava-1.5-7b-hf:
  resize shortest edge to 336 (aspect preserved), then centre-crop 336x336.
  Patch boxes are mapped back to original image coordinates and measured there.
  """
  height, width = mask.shape
  scale = shortest_edge / min(height, width)
  new_h, new_w = int(round(height * scale)), int(round(width * scale))
  top, left = (new_h - crop) // 2, (new_w - crop) // 2
  grid = crop // patch
  cov = np.zeros((grid, grid), dtype=np.float32)
  for r in range(grid):
    y0, y1 = (top + r * patch) / scale, (top + (r + 1) * patch) / scale
    ys = slice(max(0, int(np.floor(y0))), min(height, int(np.ceil(y1))))
    for c in range(grid):
      x0, x1 = (left + c * patch) / scale, (left + (c + 1) * patch) / scale
      xs = slice(max(0, int(np.floor(x0))), min(width, int(np.ceil(x1))))
      region = mask[ys, xs]
      cov[r, c] = region.mean() if region.size else 0.0
  return cov


def select_samples(ann_path, n_images, min_area, max_area, min_object_patches,
                   coverage_threshold, seed):
  """Pick one target object per image: unique instance of its class, area in range."""
  with open(ann_path, "r", encoding="utf-8") as f:
    coco = json.load(f)
  cats = {c["id"]: c["name"] for c in coco["categories"]}
  imgs = {im["id"]: im for im in coco["images"]}
  by_image = defaultdict(list)
  for ann in coco["annotations"]:
    by_image[ann["image_id"]].append(ann)

  rng = random.Random(seed)
  samples = []
  for image_id in sorted(by_image):
    anns = by_image[image_id]
    class_counts = Counter(a["category_id"] for a in anns)
    candidates = [
        a for a in anns
        if not a["iscrowd"]
        and min_area <= a["area"] <= max_area
        and class_counts[a["category_id"]] == 1
        and not isinstance(a["segmentation"], dict)
    ]
    if not candidates:
      continue
    ann = max(candidates, key=lambda a: a["area"])
    info = imgs[image_id]
    mask = ann_to_mask(ann, info["height"], info["width"])
    if mask is None or not mask.any():
      continue
    cov = patch_coverage(mask)
    object_patches = np.flatnonzero(cov.reshape(-1) >= coverage_threshold)
    if len(object_patches) < min_object_patches:
      continue  # object mostly cropped away, or too thin to cover whole patches
    samples.append({
        "image_id": image_id,
        "file_name": info["file_name"],
        "class_name": cats[ann["category_id"]],
        "area": ann["area"],
        "object_patches": object_patches.tolist(),
        "outside_patches": np.flatnonzero(cov.reshape(-1) == 0).tolist(),
    })
  rng.shuffle(samples)
  return samples[:n_images]


# --------------------------------------------------------------------------------------
# Scoring helpers
# --------------------------------------------------------------------------------------


def class_token_ids(tokenizer, name):
  """First-subtoken ids for a class name, across the casing/leading-space variants."""
  ids = set()
  for variant in (name, " " + name, name.capitalize(), " " + name.capitalize()):
    toks = tokenizer.encode(variant, add_special_tokens=False)
    if toks:
      ids.add(toks[0])
  return ids


def mentions(text, name):
  """True if the class name occurs as a word (simple plural allowed) in the text."""
  pattern = r"\b" + re.escape(name.lower()) + r"(e?s)?\b"
  return re.search(pattern, text.lower()) is not None


def first_segment(text, prompt_key):
  """The part of a generation that answers the prompt, before it starts a new example."""
  if prompt_key == "identity":
    return text.split("\n")[0]
  return re.split(r"[,\n]", text)[0]


# --------------------------------------------------------------------------------------
# Stage 1: extract visual token states + logit lens
# --------------------------------------------------------------------------------------


def get_decoder(mt):
  """LLaVA's bare LLM decoder (transformers >= 4.5x nests it under model.model)."""
  return mt.model.model.language_model


def capture_visual_states(mt, processor, image, verify=False):
  """Run one image through the VLM, returning visual token states for every layer.

  Returns a tensor of shape [num_layers, num_visual_tokens, hidden] on CPU (fp16),
  where index `l` is the OUTPUT of decoder layer `l` (pre final-norm) -- exactly the
  tensor that `set_hs_patch_hooks_llava_batch` overwrites on the target side.
  """
  inputs = processor(images=image, text=SOURCE_PROMPT, return_tensors="pt")
  inputs = {k: v.to(mt.device) for k, v in inputs.items()}
  input_ids = inputs["input_ids"][0]

  image_token_id = mt.model.config.image_token_index
  image_positions = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
  if len(image_positions) == 0:
    raise ValueError("No <image> placeholder found in the tokenized source prompt.")
  start = int(image_positions[0].item())

  layers = get_decoder(mt).layers
  captured = {}

  def make_hook(layer_idx):
    def hook(module, inp, out):
      hs = out[0] if isinstance(out, tuple) else out
      captured[layer_idx] = hs[0].detach()
    return hook

  handles = [l.register_forward_hook(make_hook(i)) for i, l in enumerate(layers)]
  try:
    with torch.no_grad():
      out = mt.model(**inputs, output_hidden_states=verify, use_cache=False)
  finally:
    remove_hooks(handles)

  n_visual = mt.num_visual_tokens
  states = torch.stack([captured[i][start:start + n_visual] for i in range(len(layers))])

  if verify:
    # Captured layer outputs must match HF's own hidden_states (except the last entry,
    # which HF reports after the final norm).
    for layer_idx in (0, len(layers) // 2, len(layers) - 2):
      hf = out.hidden_states[layer_idx + 1][0, start:start + n_visual]
      ok = torch.allclose(hf.float(), states[layer_idx].float(), atol=1e-3)
      print(f"  [verify] layer {layer_idx}: captured == hidden_states[{layer_idx + 1}]: {ok}")
    print(f"  [verify] visual tokens start at position {start}, "
          f"count in input_ids = {len(image_positions)}")

  return states.to(torch.float16).cpu()


def logit_lens(mt, states, topk=5):
  """Project layer-output states through final norm + lm_head; return top-k token ids.

  states: [n_positions, hidden] on any device. Returns ids [n_positions, topk] (CPU).
  """
  norm = get_decoder(mt).norm
  head = mt.model.lm_head
  with torch.no_grad():
    hs = states.to(mt.device, dtype=next(head.parameters()).dtype)
    logits = head(norm(hs))
    return logits.topk(topk, dim=-1).indices.cpu()


def run_extraction(mt, processor, samples, args):
  """Stage 1: cache sampled patch states and score logit lens on all object patches."""
  rng = np.random.RandomState(args.seed)
  cache_states, cache_meta, lens_rows = [], [], []

  for idx, sample in enumerate(samples):
    path = ensure_image(args.coco_dir, sample["file_name"])
    image = Image.open(path).convert("RGB")
    states = capture_visual_states(mt, processor, image, verify=(idx == 0))
    class_ids = class_token_ids(mt.tokenizer, sample["class_name"])

    # Logit lens over *all* object patches, at every layer.
    obj = np.array(sample["object_patches"])
    for layer in range(states.shape[0]):
      top = logit_lens(mt, states[layer, obj], topk=5)
      for j, patch_idx in enumerate(obj):
        ids = top[j].tolist()
        lens_rows.append({
            "image_id": sample["image_id"],
            "class_name": sample["class_name"],
            "patch_index": int(patch_idx),
            "layer": layer,
            "top1_token": mt.tokenizer.decode([ids[0]]),
            "top1_correct": ids[0] in class_ids,
            "top5_correct": any(i in class_ids for i in ids),
        })

    # Sample the patches that the (expensive) Patchscopes stage will use.
    n_obj = min(args.patches_per_image, len(sample["object_patches"]))
    chosen_obj = rng.choice(sample["object_patches"], size=n_obj, replace=False)
    outside = sample["outside_patches"]
    n_out = min(args.outside_per_image, len(outside))
    chosen_out = rng.choice(outside, size=n_out, replace=False) if n_out else []

    for patch_idx in list(chosen_obj):
      cache_meta.append({**{k: sample[k] for k in ("image_id", "file_name", "class_name")},
                         "patch_index": int(patch_idx), "kind": "object"})
      cache_states.append(states[:, int(patch_idx)])
    for patch_idx in list(chosen_out):
      cache_meta.append({**{k: sample[k] for k in ("image_id", "file_name", "class_name")},
                         "patch_index": int(patch_idx), "kind": "outside"})
      cache_states.append(states[:, int(patch_idx)])

    if (idx + 1) % 10 == 0 or idx == 0:
      print(f"  extracted {idx + 1}/{len(samples)} images")

  meta_df = pd.DataFrame(cache_meta)
  states_tensor = torch.stack(cache_states)  # [n_rows, n_layers, hidden]
  torch.save({"states": states_tensor, "meta": meta_df},
             os.path.join(args.out_dir, "patch_states.pt"))
  lens_df = pd.DataFrame(lens_rows)
  lens_df.to_csv(os.path.join(args.out_dir, "logit_lens.csv"), index=False)
  return states_tensor, meta_df, lens_df


# --------------------------------------------------------------------------------------
# Stage 2: Patchscopes generations
# --------------------------------------------------------------------------------------


def build_rows(meta_df, layers, prompt_keys, seed):
  """One row per (cached patch, layer, target prompt), plus random-vector controls."""
  rows = []
  for row_idx, meta in meta_df.iterrows():
    for layer in layers:
      for prompt_key in prompt_keys:
        rows.append({"row_idx": row_idx, "layer": layer, "prompt_key": prompt_key,
                     "kind": meta["kind"], "class_name": meta["class_name"],
                     "image_id": meta["image_id"], "patch_index": meta["patch_index"]})
  # One norm-matched random control per image (reuses the first object patch's norm).
  first_per_image = meta_df[meta_df["kind"] == "object"].groupby("image_id").head(1)
  for row_idx, meta in first_per_image.iterrows():
    for layer in layers:
      for prompt_key in prompt_keys:
        rows.append({"row_idx": row_idx, "layer": layer, "prompt_key": prompt_key,
                     "kind": "random", "class_name": meta["class_name"],
                     "image_id": meta["image_id"], "patch_index": meta["patch_index"]})
  df = pd.DataFrame(rows)
  return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def generate_patched(mt, states_tensor, batch, args, generator):
  """Run one batch of same-layer patched generations; returns list of strings."""
  tokenizer = mt.tokenizer
  prompt_key = batch["prompt_key"].iloc[0]
  prompts = [TARGET_PROMPTS[prompt_key]] * len(batch)
  inp = make_inputs(tokenizer, prompts, mt.device)
  seq_len = inp["input_ids"].shape[1]

  hidden = []
  for _, row in batch.iterrows():
    vec = states_tensor[row["row_idx"], row["layer"]].to(mt.device, torch.float16)
    if row["kind"] == "random":
      rand = torch.randn(vec.shape, generator=generator, device="cpu")
      rand = rand / rand.norm() * vec.norm().cpu()
      vec = rand.to(mt.device, torch.float16)
    hidden.append(vec)

  config = [{
      "batch_idx": i,
      "layer_target": int(row["layer"]),
      "position_target": seq_len - 1,  # last token of the target prompt
      "hidden_rep": hidden[i],
      "skip_final_ln": False,
  } for i, (_, row) in enumerate(batch.iterrows())]

  hooks = set_hs_patch_hooks_llava_batch(
      mt.model, config, module="hs", patch_input=False, generation_mode=True)
  try:
    with torch.no_grad():
      out = mt.model.generate(
          input_ids=inp["input_ids"],
          attention_mask=inp["attention_mask"],
          max_new_tokens=args.max_gen_len,
          do_sample=False,
          pad_token_id=tokenizer.eos_token_id,
      )[:, seq_len:]
  finally:
    remove_hooks(hooks)
  return [tokenizer.decode(o, skip_special_tokens=True) for o in out]


def run_patchscopes(mt, states_tensor, meta_df, args, layers):
  """Stage 2: batched patched generation for every (patch, layer, prompt) row."""
  rows = build_rows(meta_df, layers, args.prompts, args.seed)
  generator = torch.Generator().manual_seed(args.seed)
  generations = [None] * len(rows)

  # Batch within a single prompt so every sequence in a batch has the same length.
  for prompt_key in args.prompts:
    subset = rows.index[rows["prompt_key"] == prompt_key].tolist()
    for start in range(0, len(subset), args.batch_size):
      idxs = subset[start:start + args.batch_size]
      texts = generate_patched(mt, states_tensor, rows.loc[idxs], args, generator)
      for i, text in zip(idxs, texts):
        generations[i] = text
      done = start + len(idxs)
      if done % (args.batch_size * 20) == 0 or done == len(subset):
        print(f"  [{prompt_key}] {done}/{len(subset)} generations")

  rows["generation"] = generations
  rows["correct"] = [
      mentions(g, c) for g, c in zip(rows["generation"], rows["class_name"])
  ]
  rows["correct_strict"] = [
      mentions(first_segment(g, p), c)
      for g, p, c in zip(rows["generation"], rows["prompt_key"], rows["class_name"])
  ]
  rows.to_csv(os.path.join(args.out_dir, "patchscopes.csv"), index=False)
  return rows


def run_no_injection_baseline(mt, meta_df, args):
  """What the target prompts generate with no patch at all."""
  tokenizer = mt.tokenizer
  records = []
  for prompt_key in args.prompts:
    inp = make_inputs(tokenizer, [TARGET_PROMPTS[prompt_key]], mt.device)
    seq_len = inp["input_ids"].shape[1]
    with torch.no_grad():
      out = mt.model.generate(
          input_ids=inp["input_ids"], attention_mask=inp["attention_mask"],
          max_new_tokens=args.max_gen_len, do_sample=False,
          pad_token_id=tokenizer.eos_token_id)[:, seq_len:]
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    hit_rate = float(np.mean([mentions(text, c) for c in meta_df["class_name"]]))
    records.append({"prompt_key": prompt_key, "generation": text,
                    "class_mention_rate": hit_rate})
    print(f"  [no injection | {prompt_key}] {text!r} (mentions class: {hit_rate:.3f})")
  df = pd.DataFrame(records)
  df.to_csv(os.path.join(args.out_dir, "no_injection_baseline.csv"), index=False)
  return df


# --------------------------------------------------------------------------------------
# Stage 3: linear probe + summary
# --------------------------------------------------------------------------------------


def run_probe(states_tensor, meta_df, layers, args):
  """Supervised reference: logistic regression on the same cached object-patch states."""
  try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
  except ImportError:
    print("  sklearn not available -- skipping linear probe")
    return pd.DataFrame(columns=["layer", "probe_accuracy"])

  obj = meta_df[meta_df["kind"] == "object"]
  counts = obj.groupby("class_name")["image_id"].nunique()
  keep_classes = counts[counts >= args.probe_min_images].index
  obj = obj[obj["class_name"].isin(keep_classes)]
  if obj["class_name"].nunique() < 2:
    print("  too few classes with enough images -- skipping linear probe")
    return pd.DataFrame(columns=["layer", "probe_accuracy"])

  idx = obj.index.to_numpy()
  y = obj["class_name"].to_numpy()
  groups = obj["image_id"].to_numpy()
  n_splits = min(5, len(np.unique(groups)))
  records = []
  for layer in layers:
    X = states_tensor[idx, layer].float().numpy()
    preds = np.empty_like(y)
    for train, test in GroupKFold(n_splits=n_splits).split(X, y, groups):
      clf = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, multi_class="auto"))
      clf.fit(X[train], y[train])
      preds[test] = clf.predict(X[test])
    records.append({"layer": layer, "probe_accuracy": float((preds == y).mean())})
    print(f"  probe layer {layer}: {records[-1]['probe_accuracy']:.3f}")
  df = pd.DataFrame(records)
  df["n_samples"] = len(idx)
  df["n_classes"] = obj["class_name"].nunique()
  df.to_csv(os.path.join(args.out_dir, "linear_probe.csv"), index=False)
  return df


def summarize(lens_df, ps_df, probe_df, args, layers):
  """Per-layer comparison table + plot."""
  lens_layer = lens_df.groupby("layer")[["top1_correct", "top5_correct"]].mean()
  lens_layer.columns = ["logit_lens_top1", "logit_lens_top5"]

  table = lens_layer.reindex(layers)
  for prompt_key in args.prompts:
    for kind in ("object", "outside", "random"):
      subset = ps_df[(ps_df["prompt_key"] == prompt_key) & (ps_df["kind"] == kind)]
      if len(subset):
        table[f"ps_{prompt_key}_{kind}"] = subset.groupby("layer")["correct"].mean()
        if kind == "object":
          table[f"ps_{prompt_key}_object_strict"] = (
              subset.groupby("layer")["correct_strict"].mean())
  if len(probe_df):
    table["linear_probe"] = probe_df.set_index("layer")["probe_accuracy"]

  table = table.reset_index().rename(columns={"index": "layer"})
  table.to_csv(os.path.join(args.out_dir, "summary_by_layer.csv"), index=False)

  # Neo et al. report a single headline number; compute both plausible readings.
  per_image_best = (lens_df.groupby(["image_id", "layer"])["top1_correct"].mean()
                    .groupby("image_id").max().mean())
  headline = {
      "logit_lens_top1_best_layer": float(lens_layer["logit_lens_top1"].max()),
      "logit_lens_best_layer": int(lens_layer["logit_lens_top1"].idxmax()),
      "logit_lens_top1_per_image_best_layer_mean": float(per_image_best),
      "n_images": int(lens_df["image_id"].nunique()),
      "n_object_patches": int(len(lens_df) / lens_df["layer"].nunique()),
  }
  for prompt_key in args.prompts:
    col = f"ps_{prompt_key}_object"
    if col in table:
      headline[f"{col}_best"] = float(table[col].max())
      headline[f"{col}_best_layer"] = int(table.loc[table[col].idxmax(), "layer"])
  with open(os.path.join(args.out_dir, "headline.json"), "w", encoding="utf-8") as f:
    json.dump(headline, f, indent=2)

  try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for col in table.columns:
      if col == "layer":
        continue
      style = "--" if ("outside" in col or "random" in col) else "-"
      ax.plot(table["layer"], table[col], style, marker="o", ms=3, label=col)
    ax.set_xlabel("LLM backbone layer")
    ax.set_ylabel("object identification accuracy")
    ax.set_title("Reading the object out of a visual patch token (LLaVA-1.5-7B, COCO)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "object_identification.png"), dpi=150)
  except ImportError:
    print("  matplotlib not available -- skipping plot")

  return table, headline


def save_debug_overlays(processor, samples, args, n=5):
  """Save the *processed* 336x336 image with object patches outlined, to check geometry."""
  out_dir = os.path.join(args.out_dir, "overlays")
  os.makedirs(out_dir, exist_ok=True)
  mean = np.array(processor.image_processor.image_mean)
  std = np.array(processor.image_processor.image_std)
  for sample in samples[:n]:
    path = ensure_image(args.coco_dir, sample["file_name"])
    pixel_values = processor(images=Image.open(path).convert("RGB"),
                             text=SOURCE_PROMPT, return_tensors="pt")["pixel_values"][0]
    arr = pixel_values.permute(1, 2, 0).numpy() * std + mean
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    for patch_idx in sample["object_patches"]:
      r, c = divmod(patch_idx, 24)
      draw.rectangle([c * 14, r * 14, (c + 1) * 14 - 1, (r + 1) * 14 - 1], outline=(255, 0, 0))
    img.save(os.path.join(out_dir, f"{sample['image_id']}_{sample['class_name']}.png"))
  print(f"  wrote overlays to {out_dir}")


# --------------------------------------------------------------------------------------


def parse_args(argv=None):
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--coco_dir", required=True,
                 help="directory holding annotations/ and val2017/ (downloaded if absent)")
  p.add_argument("--out_dir", default="./results/object_identification")
  p.add_argument("--model_name", default="llava-hf/llava-1.5-7b-hf")
  p.add_argument("--n_images", type=int, default=170)
  p.add_argument("--min_area", type=float, default=20000)
  p.add_argument("--max_area", type=float, default=30000)
  p.add_argument("--coverage_threshold", type=float, default=0.5,
                 help="patch counts as an object patch above this mask fraction")
  p.add_argument("--patches_per_image", type=int, default=4)
  p.add_argument("--outside_per_image", type=int, default=2)
  p.add_argument("--layer_step", type=int, default=1)
  p.add_argument("--batch_size", type=int, default=64)
  p.add_argument("--max_gen_len", type=int, default=20)
  p.add_argument("--prompts", nargs="+", default=["identity", "entity"],
                 choices=sorted(TARGET_PROMPTS))
  p.add_argument("--probe_min_images", type=int, default=3)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--debug_overlays", action="store_true")
  p.add_argument("--overwrite", action="store_true")
  return p.parse_args(argv)


def main(argv=None):
  args = parse_args(argv)
  os.makedirs(args.out_dir, exist_ok=True)

  print("Selecting COCO samples ...")
  ann_path = ensure_annotations(args.coco_dir)
  samples_path = os.path.join(args.out_dir, "samples.json")
  if os.path.exists(samples_path) and not args.overwrite:
    with open(samples_path, "r", encoding="utf-8") as f:
      samples = json.load(f)
  else:
    samples = select_samples(ann_path, args.n_images, args.min_area, args.max_area,
                             args.patches_per_image, args.coverage_threshold, args.seed)
    with open(samples_path, "w", encoding="utf-8") as f:
      json.dump(samples, f, indent=1)
  print(f"  {len(samples)} images, {len(set(s['class_name'] for s in samples))} classes, "
        f"{sum(len(s['object_patches']) for s in samples)} object patches")

  print(f"Loading {args.model_name} ...")
  mt = ModelAndTokenizer(args.model_name, torch_dtype=torch.float16, device="cuda")
  processor = mt.processor
  layers = list(range(0, mt.num_layers, args.layer_step))
  print(f"  num_layers={mt.num_layers}, evaluating layers {layers}")

  if args.debug_overlays:
    save_debug_overlays(processor, samples, args)

  cache_path = os.path.join(args.out_dir, "patch_states.pt")
  if os.path.exists(cache_path) and not args.overwrite:
    print("Stage 1: reusing cached states")
    cache = torch.load(cache_path, weights_only=False)
    states_tensor, meta_df = cache["states"], cache["meta"]
    lens_df = pd.read_csv(os.path.join(args.out_dir, "logit_lens.csv"))
  else:
    print("Stage 1: extracting visual token states + logit lens ...")
    states_tensor, meta_df, lens_df = run_extraction(mt, processor, samples, args)

  print("Stage 2: Patchscopes generations ...")
  run_no_injection_baseline(mt, meta_df, args)
  ps_df = run_patchscopes(mt, states_tensor, meta_df, args, layers)

  print("Stage 3: linear probe ...")
  probe_df = run_probe(states_tensor, meta_df, layers, args)

  table, headline = summarize(lens_df, ps_df, probe_df, args, layers)
  print("\n=== per-layer accuracy ===")
  print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
  print("\n=== headline ===")
  print(json.dumps(headline, indent=2))
  print(f"\nAll outputs in {args.out_dir}")


if __name__ == "__main__":
  main()
