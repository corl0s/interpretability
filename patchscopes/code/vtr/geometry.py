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
  grid = 24
  n_tokens = 576

  def coverage(self, mask):
    return patch_coverage(mask).reshape(-1)


GEOMETRIES = {"llava-1.5": Llava15Geometry}


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
