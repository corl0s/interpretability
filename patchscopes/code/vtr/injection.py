"""Injection readouts designed for visual tokens (prototype of the Patchscopes redesign).

A visual state injected into a text placeholder lands somewhere the model never sees such states.
These variants instead give the model the state where it expects visual information, and control
its scale and persistence:

  target "text"   SelfIE-style text prompt; the state replaces the five placeholder states
  target "image"  LLaVA chat prompt with a neutral (mean-colour) image and a question; the state
                  replaces image-token states at chosen grid positions:
                    all        every image token
                    own        the token's own grid position (pooled / random rows: footprint)
                    own3x3     the 3x3 block around it
                    footprint  every token the object covers (coverage >= 0.5)
  norm_match      rescale the state to the median norm of the states it replaces at that layer
  extra_layers    also re-inject at target + each offset (the state is not washed out)
  add             add the state to the slot states instead of replacing them

Scoring is first-token closed-set: a class scores the best log-probability of the first token of
" <form>" after the prompt, raw and PMI (minus the same with no injection). That needs one
forward pass per state, so ~600-token image prompts stay cheap. On the fair subset no two
classes share a first sub-token, so ranks there are unaffected by the approximation.
"""

import numpy as np
import torch
from PIL import Image

from visual_object_identification import get_decoder
from vtr.geometry import HIGH_COVERAGE
from vtr.models import text_position_ids
from vtr.readouts import PROMPTS, _scatter_max, placeholder_positions

IMAGE_QUESTION = "What is the main object in this image? Answer with one word."
SLOTS = ("all", "own", "own3x3", "footprint")


class InjectionReadout:

  def __init__(self, mt, class_names, forms, target, device, question=IMAGE_QUESTION,
               rows_per_batch=16):
    self.mt, self.device, self.target = mt, device, target
    self.layers = get_decoder(mt).layers
    self.batch = rows_per_batch
    tok = mt.tokenizer
    if target == "text":
      prompt, continuation, placeholder = PROMPTS["selfie"]
      self.inputs = {"input_ids": torch.tensor([tok(prompt)["input_ids"]])}
      self.pos = placeholder_positions(tok, self.inputs["input_ids"][0].tolist(), placeholder,
                                       "selfie")
    elif target == "image":
      if getattr(mt, "family", "llava") != "llava":
        raise NotImplementedError("image-slot injection is implemented for LLaVA only")
      prompt, continuation = f"USER: <image>\n{question} ASSISTANT:", " {}"
      mean = tuple(int(255 * m) for m in mt.processor.image_processor.image_mean)
      grey = Image.new("RGB", (336, 336), mean)
      self.inputs = dict(mt.processor(images=grey, text=prompt, return_tensors="pt"))
      ids = self.inputs["input_ids"][0]
      self.pos = (ids == mt.model.config.image_token_index).nonzero(as_tuple=True)[0].tolist()
      self.side = int(round(len(self.pos) ** 0.5))
      if self.side ** 2 != len(self.pos) or self.pos[-1] - self.pos[0] + 1 != len(self.pos):
        raise ValueError(f"expected a contiguous square image grid, found {len(self.pos)} tokens")
    else:
      raise ValueError(target)
    self.P = self.inputs["input_ids"].shape[1]

    # First token of each class form's continuation (the prompt text is tokenized alone, with
    # <image> as one token, so continuation tokens are unaffected by the image expansion).
    base = tok(prompt)["input_ids"]
    first, owners = [], []
    for ci, c in enumerate(class_names):
      for f in forms[c]:
        full = tok(prompt + continuation.format(f))["input_ids"]
        if full[:len(base)] != base:
          raise ValueError(f"prompt tokenization changes when followed by {f!r}")
        first.append(full[len(base)])
        owners.append(ci)
    self.first = torch.tensor(first)
    self.owner = torch.tensor(owners)
    self.n_classes = len(class_names)
    self.ref_norm = self._reference_norms()
    self.prior = self._logp(None, None, None)[0]  # [S], prompt alone

  def slots(self, mode, token=-1, coverage=None):
    """Prompt positions to overwrite for one row."""
    if self.target == "text" or mode == "placeholders":
      return list(self.pos)
    grid = None
    if mode == "all":
      grid = range(self.side ** 2)
    elif mode in ("own", "own3x3") and token >= 0:
      r, c = divmod(int(token), self.side)
      k = 0 if mode == "own" else 1
      grid = [rr * self.side + cc for rr in range(max(0, r - k), min(self.side, r + k + 1))
              for cc in range(max(0, c - k), min(self.side, c + k + 1))]
    elif mode in ("own", "own3x3", "footprint"):
      grid = np.flatnonzero(np.asarray(coverage) >= HIGH_COVERAGE)
    else:
      raise ValueError(mode)
    return [self.pos[i] for i in grid]

  def _batch_inputs(self, n):
    out = {k: v.repeat(n, *([1] * (v.dim() - 1))).to(self.device) for k, v in self.inputs.items()}
    out["attention_mask"] = torch.ones_like(out["input_ids"])
    if self.target == "text":
      out["position_ids"] = text_position_ids(self.mt, 0, self.P, n, self.device)
    return out

  @torch.no_grad()
  def _reference_norms(self):
    """Median norm of the clean states at the injectable positions, per layer."""
    norms = {}
    handles = [l.register_forward_hook(
        lambda m, i, o, k=k: norms.__setitem__(
            k, (o[0] if isinstance(o, tuple) else o)[0, self.pos].float().norm(dim=-1).median()
            .item()))
        for k, l in enumerate(self.layers)]
    try:
      self.mt.model(**self._batch_inputs(1), use_cache=False)
    finally:
      for h in handles:
        h.remove()
    return torch.tensor([norms[k] for k in range(len(self.layers))])

  def _logp(self, states, positions, layer, norm_match=False, extra_layers=(), add=False):
    """First-token log-prob of every class form after injection. -> [n, S]."""
    lp = self.next_logprobs(states, positions, layer, norm_match, extra_layers, add)
    return lp[:, self.first.to(self.device)].cpu()

  @torch.no_grad()
  def next_logprobs(self, states, positions, layer, norm_match=False, extra_layers=(), add=False):
    """Next-token log-probs over the whole vocabulary after injection. -> [n, V] on device."""
    n = 1 if states is None else states.shape[0]
    handles = []
    if states is not None:
      states = states.to(self.device).float()
      width = max(len(p) for p in positions)
      pos = torch.tensor([p + [p[0]] * (width - len(p)) for p in positions], device=self.device)
      rows = torch.arange(n, device=self.device)[:, None].expand_as(pos)
      unit = states / states.norm(dim=-1, keepdim=True).clamp_min(1e-6)
      for l in sorted({layer, *(layer + o for o in extra_layers)}):
        if l >= len(self.layers):
          continue
        vec = unit * float(self.ref_norm[l]) if norm_match else states

        def hook(module, inp, out, vec=vec):
          hs = out[0] if isinstance(out, tuple) else out
          v = vec.to(hs.dtype)[:, None, :].expand(-1, pos.shape[1], -1)
          hs[rows, pos] = hs[rows, pos] + v if add else v
        handles.append(self.layers[l].register_forward_hook(hook))
    try:
      out = self.mt.model(**self._batch_inputs(n), use_cache=False)
    finally:
      for h in handles:
        h.remove()
    return out.logits[:, -1].float().log_softmax(-1)

  def scores(self, h, layer, positions, norm_match=False, extra_layers=(), add=False):
    """h [n, d]; positions: one list of prompt positions per row. Returns (raw, pmi)."""
    raw, pmi = [], []
    for start in range(0, h.shape[0], self.batch):
      s = self._logp(h[start:start + self.batch], positions[start:start + self.batch], layer,
                     norm_match, extra_layers, add)
      raw.append(_scatter_max(s, self.owner, self.n_classes))
      pmi.append(_scatter_max(s - self.prior, self.owner, self.n_classes))
    return torch.cat(raw).numpy(), torch.cat(pmi).numpy()
