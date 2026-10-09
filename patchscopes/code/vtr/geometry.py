"""Map a model's visual tokens to the image region each one covers.

Only LLaVA-1.5 is implemented in Milestone 1. Each geometry turns an object mask (original
image coordinates) into a per-token coverage vector: the fraction of the token's region that the
object covers. Everything downstream (overlap bins, token sampling, pooling) uses that vector,
so adding LLaVA-NeXT or Qwen2.5-VL only needs a new class with the same `coverage` method.
"""

import numpy as np

from visual_object_identification import patch_coverage

# Overlap bins for object tokens; coverage == 0 is "outside".
BINS = (("0-25", 0.0, 0.25), ("25-50", 0.25, 0.5), ("50-75", 0.5, 0.75), ("75-100", 0.75, 1.0))
HIGH_COVERAGE = 0.5  # tokens at or above this count as "on the object" (pooling, probe training)


class Llava15Geometry:
  """CLIP 336px: resize shortest edge to 336, centre-crop 336x336, 24x24 patches of 14px."""

  name = "llava-1.5"
  n_tokens = 576

  def grid(self, height, width):
    return 24, 24

  def coverage(self, mask):
    return patch_coverage(mask).reshape(-1)


def qwen_smart_resize(height, width, factor=28, min_pixels=56 * 56,
                      max_pixels=14 * 14 * 4 * 1280):
  """Qwen2-VL / Qwen2.5-VL resize rule (transformers image_processing_qwen2_vl.smart_resize)."""
  import math
  h_bar = round(height / factor) * factor
  w_bar = round(width / factor) * factor
  if h_bar * w_bar > max_pixels:
    beta = math.sqrt((height * width) / max_pixels)
    h_bar = max(factor, math.floor(height / beta / factor) * factor)
    w_bar = max(factor, math.floor(width / beta / factor) * factor)
  elif h_bar * w_bar < min_pixels:
    beta = math.sqrt(min_pixels / (height * width))
    h_bar = math.ceil(height * beta / factor) * factor
    w_bar = math.ceil(width * beta / factor) * factor
  return h_bar, w_bar


class Qwen25Geometry:
  """Qwen2.5-VL: aspect-preserving resize to multiples of 28 px (no crop); each LLM token is a
  2x2 merge of 14-px patches, i.e. a 28x28 cell of the resized image, in row-major order.

  The token grid depends on the image size, so `grid(h, w)` gives (rows, cols) per image.
  """

  name = "qwen2.5-vl"
  patch = 14
  merge = 2

  def __init__(self, min_pixels=56 * 56, max_pixels=12845056):
    self.min_pixels, self.max_pixels = min_pixels, max_pixels

  @property
  def cell(self):
    return self.patch * self.merge

  def resized(self, height, width):
    return qwen_smart_resize(height, width, self.cell, self.min_pixels, self.max_pixels)

  def grid(self, height, width):
    h_bar, w_bar = self.resized(height, width)
    return h_bar // self.cell, w_bar // self.cell

  def coverage(self, mask):
    """Fraction of each token's cell covered by the mask, flattened row-major."""
    height, width = mask.shape
    h_bar, w_bar = self.resized(height, width)
    rows, cols = h_bar // self.cell, w_bar // self.cell
    sy, sx = height / h_bar, width / w_bar
    cov = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
      ys = slice(int(np.floor(r * self.cell * sy)), min(height, int(np.ceil((r + 1) * self.cell * sy))))
      for c in range(cols):
        xs = slice(int(np.floor(c * self.cell * sx)), min(width, int(np.ceil((c + 1) * self.cell * sx))))
        region = mask[ys, xs]
        cov[r, c] = region.mean() if region.size else 0.0
    return cov.reshape(-1)


GEOMETRIES = {"llava-1.5": Llava15Geometry, "qwen2.5-vl": Qwen25Geometry}


def coverage_bin(c):
  """Bin label for a coverage value; 'outside' for exactly 0."""
  if c <= 0:
    return "outside"
  for name, lo, hi in BINS:
    if lo < c <= hi:
      return name
  return BINS[-1][0]


def sample_tokens(coverage, per_bin, n_outside, rng):
  """Stratified token sample: up to `per_bin` object tokens per overlap bin + outside tokens.

  Returns a list of (token_index, bin_label).
  """
  coverage = np.asarray(coverage)
  picks = []
  for name, lo, hi in BINS:
    idx = np.flatnonzero((coverage > lo) & (coverage <= hi))
    if len(idx):
      chosen = rng.choice(idx, size=min(per_bin, len(idx)), replace=False)
      picks += [(int(i), name) for i in sorted(chosen)]
  outside = np.flatnonzero(coverage == 0)
  if len(outside) and n_outside:
    chosen = rng.choice(outside, size=min(n_outside, len(outside)), replace=False)
    picks += [(int(i), "outside") for i in sorted(chosen)]
  return picks
