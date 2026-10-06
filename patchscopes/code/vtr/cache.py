"""Capture the hidden states the readouts will decode, for every layer.

Per image, rows of kind:
  object       -- sampled object tokens, stratified by overlap bin (coverage recorded)
  outside      -- tokens the object does not touch (control)
  pooled_mean  -- mean over all tokens with coverage >= 0.5 ("increasing accessibility")
  pooled_max   -- element-wise max over the same tokens
  random       -- Gaussian vector, norm-matched per layer to pooled_mean (control)

States are the output of decoder block l (pre final-norm), as everywhere in this project.
"""

import numpy as np
import pandas as pd
import torch
from PIL import Image

from visual_object_identification import capture_visual_states, ensure_image
from vtr.geometry import HIGH_COVERAGE, sample_tokens


def cache_states(mt, processor, samples, coco_dir, per_bin, n_outside, seed):
  """Returns (states [n_rows, n_layers, hidden] fp16 CPU, meta DataFrame)."""
  rng = np.random.RandomState(seed)
  gen = torch.Generator().manual_seed(seed)
  states, meta = [], []

  for idx, sample in enumerate(samples):
    image = Image.open(ensure_image(coco_dir, sample["file_name"])).convert("RGB")
    full = capture_visual_states(mt, processor, image, verify=(idx == 0))  # [L, T, d]
    cov = np.asarray(sample["coverage"])
    base = {k: sample[k] for k in ("image_id", "class_name", "supercategory", "size")}

    for token, bin_label in sample_tokens(cov, per_bin, n_outside, rng):
      states.append(full[:, token])
      meta.append({**base, "kind": "object" if bin_label != "outside" else "outside",
                   "token": token, "coverage": float(cov[token]), "bin": bin_label})

    on_object = torch.as_tensor(np.flatnonzero(cov >= HIGH_COVERAGE))
    obj = full[:, on_object].float()                       # [L, k, d]
    pooled_mean = obj.mean(dim=1)
    pooled_max = obj.max(dim=1).values
    for kind, vec in (("pooled_mean", pooled_mean), ("pooled_max", pooled_max)):
      states.append(vec.half())
      meta.append({**base, "kind": kind, "token": -1, "coverage": np.nan, "bin": kind})

    noise = torch.randn(pooled_mean.shape, generator=gen)
    noise = noise / noise.norm(dim=-1, keepdim=True) * pooled_mean.norm(dim=-1, keepdim=True)
    states.append(noise.half())
    meta.append({**base, "kind": "random", "token": -1, "coverage": np.nan, "bin": "random"})

    if (idx + 1) % 25 == 0 or idx == 0:
      print(f"  cached {idx + 1}/{len(samples)} images ({len(meta)} rows)")

  return torch.stack(states), pd.DataFrame(meta)
