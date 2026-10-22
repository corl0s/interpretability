"""Visual activation oracle: LLaVA with LoRA, trained to name what an injected visual state encodes.

Self-supervised: the training target for a visual token is LLaVA's own answer about the image
region of that token (a red box drawn around its patch block), so no class label is used. The
oracle reads the state the way image-slot injection does (vtr/injection.py): a neutral-image
prompt whose image-token states are replaced by the (norm-matched) state at one target layer.

  LoRALinear      low-rank adapter around a frozen nn.Linear (no peft dependency)
  add_lora        wrap the attention projections of every decoder layer
  square_crop     CLIP 336 geometry (resize shortest edge, centre crop), so the token grid matches
  draw_box        red box around a token's patch block on the 336x336 image
  clean_answer    normalise a generated one-word answer
  slot_injection  forward hook that writes one state per row into the image slots (differentiable)
"""

import math
import re
from contextlib import contextmanager

import torch
import torch.nn as nn
from PIL import Image, ImageDraw

from visual_object_identification import get_decoder

BOX_QUESTION = "What is the object inside the red box? Answer with one word."
# Answers about the box itself, or none at all, are not training targets.
BAD_ANSWERS = {"", "box", "red box", "square", "rectangle", "red", "frame", "border", "outline",
               "nothing", "none", "unknown"}
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


class LoRALinear(nn.Module):

  def __init__(self, base, rank, alpha):
    super().__init__()
    self.base = base
    dev = base.weight.device
    self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=dev))
    self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=dev))
    nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
    self.scale = alpha / rank
    self.enabled = True

  def forward(self, x):
    y = self.base(x)
    if not self.enabled:
      return y
    return y + ((x.float() @ self.lora_A.T) @ self.lora_B.T * self.scale).to(y.dtype)


def add_lora(mt, rank, alpha, targets=LORA_TARGETS):
  for layer in get_decoder(mt).layers:
    attn = layer.self_attn
    for name in targets:
      setattr(attn, name, LoRALinear(getattr(attn, name), rank, alpha))
  return lora_parameters(mt)


def lora_parameters(mt):
  return {n: p for n, p in mt.model.named_parameters() if ".lora_" in n}


def set_lora(mt, enabled):
  for m in mt.model.modules():
    if isinstance(m, LoRALinear):
      m.enabled = enabled


def square_crop(image, size=336):
  """Same geometry as the CLIP processor and patch_coverage: resize shortest edge, centre crop."""
  w, h = image.size
  scale = size / min(w, h)
  nw, nh = int(round(w * scale)), int(round(h * scale))
  image = image.resize((nw, nh), Image.BICUBIC)
  left, top = (nw - size) // 2, (nh - size) // 2
  return image.crop((left, top, left + size, top + size))


def draw_box(image, token, side=24, patches=3, width=3):
  """Red box around the patches x patches block centred on `token` (clipped at the border)."""
  out = image.copy()
  cell = image.size[0] / side
  r, c = divmod(int(token), side)
  k = patches // 2
  x0, y0 = max(0, c - k) * cell, max(0, r - k) * cell
  x1, y1 = min(side, c + k + 1) * cell, min(side, r + k + 1) * cell
  ImageDraw.Draw(out).rectangle([x0, y0, x1 - 1, y1 - 1], outline=(255, 0, 0), width=width)
  return out


def clean_answer(text, max_words=3):
  text = text.strip().lower().split("\n")[0]
  text = re.sub(r"[^a-z\s-]", " ", text)
  words = [w for w in text.split() if w not in ("a", "an", "the")]
  return " ".join(words[:max_words])


@contextmanager
def slot_injection(mt, layer, positions, vecs):
  """Replace the output of decoder block `layer` at `positions` with vecs [B, d] (row-wise)."""
  pos = torch.as_tensor(positions, device=vecs.device)

  def hook(module, inp, out):
    hs = out[0] if isinstance(out, tuple) else out
    hs = hs.clone()
    hs[:, pos] = vecs.to(hs.dtype)[:, None, :]
    return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs

  handle = get_decoder(mt).layers[layer].register_forward_hook(hook)
  try:
    yield
  finally:
    handle.remove()
