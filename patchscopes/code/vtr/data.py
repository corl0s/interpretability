"""COCO-VTR object task: select val2017 images and their target object.

One target object per image: non-crowd polygon annotation, the only instance of its class in the
image (unambiguous label), area within [min_area, max_area], and at least `min_tokens` tokens
with coverage >= 0.5 after the model's preprocessing. Images are sampled round-robin across
classes (at most `per_class` each) so that no class dominates.
"""

import json
import random
from collections import Counter, defaultdict

import numpy as np

from visual_object_identification import ann_to_mask
from vtr.geometry import HIGH_COVERAGE

# COCO's own small / medium / large thresholds (in original-image pixels).
SIZE_BUCKETS = (("small", 32 ** 2), ("medium", 96 ** 2))


def size_bucket(area):
  for name, upper in SIZE_BUCKETS:
    if area < upper:
      return name
  return "large"


def load_coco(ann_path):
  with open(ann_path, "r", encoding="utf-8") as f:
    coco = json.load(f)
  cats = {c["id"]: c for c in coco["categories"]}
  class_names = [cats[i]["name"] for i in sorted(cats)]
  return coco, cats, class_names


def select_object_samples(ann_path, geometry, n_images, per_class, min_area, max_area,
                          min_tokens, seed):
  """Return (samples, class_names). Each sample stores per-token coverage for `geometry`."""
  coco, cats, class_names = load_coco(ann_path)
  imgs = {im["id"]: im for im in coco["images"]}
  by_image = defaultdict(list)
  for ann in coco["annotations"]:
    by_image[ann["image_id"]].append(ann)

  candidates = defaultdict(list)  # class name -> samples
  for image_id in sorted(by_image):
    anns = by_image[image_id]
    counts = Counter(a["category_id"] for a in anns)
    pool = [a for a in anns
            if not a["iscrowd"] and counts[a["category_id"]] == 1
            and not isinstance(a["segmentation"], dict)
            and min_area <= a["area"] <= (max_area or float("inf"))]
    if not pool:
      continue
    ann = max(pool, key=lambda a: a["area"])
    info = imgs[image_id]
    mask = ann_to_mask(ann, info["height"], info["width"])
    if mask is None or not mask.any():
      continue
    cov = geometry.coverage(mask)
    if int((cov >= HIGH_COVERAGE).sum()) < min_tokens:
      continue
    cat = cats[ann["category_id"]]
    candidates[cat["name"]].append({
        "image_id": image_id,
        "file_name": info["file_name"],
        "class_name": cat["name"],
        "supercategory": cat.get("supercategory", ""),
        "ann_id": ann["id"],
        "area": float(ann["area"]),
        "size": size_bucket(ann["area"]),
        "coverage": [round(float(c), 4) for c in cov],
        "grid": list(geometry.grid(info["height"], info["width"])),
    })

  rng = random.Random(seed)
  for lst in candidates.values():
    rng.shuffle(lst)
  samples, depth = [], 0
  while len(samples) < n_images and depth < per_class:
    added = False
    for name in sorted(candidates):
      if depth < len(candidates[name]) and len(samples) < n_images:
        samples.append(candidates[name][depth])
        added = True
    if not added:
      break
    depth += 1
  rng.shuffle(samples)
  return samples, class_names


def select_same_images(previous, ann_path, geometry, min_tokens):
  """Reuse another run's images and target objects, recomputing coverage for `geometry`.

  Keeps the images whose object still has >= min_tokens tokens at coverage >= 0.5 under the new
  geometry, so two models are compared on the same images and objects.
  """
  coco, _, class_names = load_coco(ann_path)
  anns = {a["id"]: a for a in coco["annotations"]}
  imgs = {im["id"]: im for im in coco["images"]}
  samples = []
  for prev in previous:
    ann, info = anns[prev["ann_id"]], imgs[prev["image_id"]]
    mask = ann_to_mask(ann, info["height"], info["width"])
    cov = geometry.coverage(mask)
    if int((cov >= HIGH_COVERAGE).sum()) < min_tokens:
      continue
    samples.append({**{k: prev[k] for k in ("image_id", "file_name", "class_name",
                                             "supercategory", "ann_id", "area", "size")},
                    "coverage": [round(float(c), 4) for c in cov],
                    "grid": list(geometry.grid(info["height"], info["width"]))})
  return samples, class_names


def summarize_samples(samples):
  classes = Counter(s["class_name"] for s in samples)
  sizes = Counter(s["size"] for s in samples)
  covs = np.array([c for s in samples for c in s["coverage"]])
  return {
      "n_images": len(samples),
      "n_classes": len(classes),
      "images_per_class": {"min": min(classes.values()), "max": max(classes.values())},
      "sizes": dict(sizes),
      "object_tokens_per_image": float((covs > 0).sum() / max(len(samples), 1)),
  }
