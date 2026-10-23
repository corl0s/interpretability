"""Visual activation oracle: LLaVA with LoRA, trained to name what an injected visual state encodes.

Self-supervised: the training target for a visual token is LLaVA's own answer about the image
region of that token (a red box drawn around its patch block), so no class label is used. The
oracle reads the state the way image-slot injection does (vtr/injection.py): a neutral-image
prompt whose image-token states are replaced by the (norm-matched) state at one target layer.

  LoRALinear      low-rank adapter around a frozen nn.Linear (no peft dependency)
  add_lora        wrap the attention projections of every decoder layer
  square_crop     CLIP 336 geometry (resize shortest edge, centre crop), so the token grid matches
  draw_box        red box around a token's patch block on the 336x336 image
  make_view       one way of pointing LLaVA at a token's region (box / crop / mask), with question
  ask             greedy one-word answers and first-step log-probs for a batch of images
  class_scores    first-step log-probs -> closed-set class scores (best first sub-token of a form)
  clean_answer    normalise a generated one-word answer
  slot_injection  forward hook that writes one state per row into the image slots (differentiable)
"""

import math
import re
from contextlib import contextmanager

import torch
import torch.nn as nn
from PIL import Image, ImageDraw

from visual_object_identification import class_token_ids, get_decoder
from vtr.injection import IMAGE_QUESTION

BOX_QUESTION = "What is the object inside the red box? Answer with one word."
GREY = (122, 117, 104)  # CLIP pixel mean
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
  """Fresh adapters (already wrapped layers get new adapters around their frozen base)."""
  for layer in get_decoder(mt).layers:
    attn = layer.self_attn
    for name in targets:
      base = getattr(attn, name)
      base = base.base if isinstance(base, LoRALinear) else base
      setattr(attn, name, LoRALinear(base, rank, alpha))
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


def block(token, k, side=24):
  """Patch-index bounds [r0, r1) x [c0, c1) of the k x k block centred on token (clipped)."""
  r, c = divmod(int(token), side)
  h = k // 2
  return max(0, r - h), min(side, r + h + 1), max(0, c - h), min(side, c + h + 1)


def make_view(image, token, view, side=24):
  """(PIL image, question) for a view name:

  full   the whole image ("main object"; the token plays no role)
  boxK   red box around the KxK patch block ("object inside the red box")
  cropK  only the KxK block, enlarged to the image size ("main object")
  maskK  the whole image with everything outside the KxK block greyed out ("main object")
  """
  if view == "full":
    return image, IMAGE_QUESTION
  kind = view.rstrip("0123456789")
  k = int(view[len(kind):])
  cell = image.size[0] / side
  r0, r1, c0, c1 = block(token, k, side)
  box = (int(c0 * cell), int(r0 * cell), int(c1 * cell), int(r1 * cell))
  if kind == "box":
    return draw_box(image, token, side=side, patches=k), BOX_QUESTION
  if kind == "crop":
    return image.crop(box).resize(image.size, Image.BICUBIC), IMAGE_QUESTION
  if kind == "mask":
    out = Image.new("RGB", image.size, GREY)
    out.paste(image.crop(box), box[:2])
    return out, IMAGE_QUESTION
  raise ValueError(f"unknown view {view!r}")


@torch.no_grad()
def ask(mt, images, question, max_new_tokens=6):
  """Greedy cleaned answers and the first-step log-probs [n, vocab] (CPU, fp32)."""
  tok = mt.tokenizer
  prompt = f"USER: <image>\n{question} ASSISTANT:"
  inputs = mt.processor(images=images, text=[prompt] * len(images), return_tensors="pt")
  inputs = {k: v.to(mt.device) for k, v in inputs.items()}
  if "pixel_values" in inputs:
    inputs["pixel_values"] = inputs["pixel_values"].to(mt.model.dtype)
  pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
  out = mt.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                          pad_token_id=pad, output_scores=True, return_dict_in_generate=True)
  new = out.sequences[:, inputs["input_ids"].shape[1]:]
  texts = tok.batch_decode(new, skip_special_tokens=True)
  return [clean_answer(t) for t in texts], out.scores[0].float().log_softmax(-1).cpu()


def first_tokens(tokenizer, class_names, forms):
  """(class index, token id) tensors over the first sub-tokens of every form of every class."""
  pairs = sorted({(ci, t) for ci, c in enumerate(class_names) for f in forms[c]
                  for t in class_token_ids(tokenizer, f)})
  return torch.tensor([p[0] for p in pairs]), torch.tensor([p[1] for p in pairs])


def class_scores(logp, cls, ids, n_classes):
  """logp [n, V] -> [n, n_classes]: best log-prob of a first sub-token of any form."""
  out = torch.full((logp.shape[0], n_classes), float("-inf"))
  return out.scatter_reduce(1, cls.expand(logp.shape[0], -1), logp[:, ids], reduce="amax",
                            include_self=True).numpy()


def clean_answer(text, max_words=3):
  text = text.strip().lower().split("\n")[0]
  text = re.sub(r"[^a-z\s-]", " ", text)
  words = [w for w in text.split() if w not in ("a", "an", "the")]
  return " ".join(words[:max_words])


@contextmanager
def slot_injection(mt, layer, positions, vecs, add=False):
  """Replace (or with add, add to) the output of decoder block `layer` at `positions` with
  vecs [B, d] (row-wise)."""
  pos = torch.as_tensor(positions, device=vecs.device)

  def hook(module, inp, out):
    hs = out[0] if isinstance(out, tuple) else out
    hs = hs.clone()
    v = vecs.to(hs.dtype)[:, None, :]
    hs[:, pos] = hs[:, pos] + v if add else v
    return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs

  handle = get_decoder(mt).layers[layer].register_forward_hook(hook)
  try:
    yield
  finally:
    handle.remove()
