"""Readouts: map hidden states at one layer to scores over the candidate classes (closed set).

Every readout returns an array [n_rows, n_classes] (higher = more likely), so all of them are
ranked against the same 80 COCO class names. A class is represented by its name and, if
`synonyms` is on, by curated synonyms that no other class shares (shared forms such as "board"
or "bag" cannot tell classes apart and are dropped).

  LogitLens     final norm + lm_head; class score = best log-prob of a first sub-token of a form
  TunedLens     LogitLens after a learned per-layer affine translator (Belrose et al., 2023);
                translators are trained on text by train_tuned_lens.py
  EmbeddingLens max cosine to the input embedding of a first sub-token of a form
  LatentLens    max cosine to bank entries whose word (in its sentence) names the class
  Patchscopes   inject the state into a text prompt; class score = log-likelihood of the
                continuation naming the class. Reported raw and prior-corrected (PMI: minus the
                same log-likelihood with no injection), because the prompt itself favours some
                words ("cat" appears in the identity prompt). The same class runs the
                SelfIE-style prompt (`selfie`), which injects at five placeholder positions.
  probe_scores  supervised linear / MLP probe, images grouped across folds (availability)
"""

import os
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

from analyze_object_identification import SURFACE_FORMS
from latentlens_object_identification import word_context, word_matches
from visual_object_identification import TARGET_PROMPTS, class_token_ids, get_decoder
from vtr.models import text_position_ids

# Injection prompts: (prompt text, how a class name continues it, placeholder token).
# placeholder None = inject at the last prompt token (Patchscopes' '?' / 'x'); otherwise inject at
# every occurrence of that token.
#  identity / entity: unmodified target prompts of the original Patchscopes notebooks.
#  selfie: SelfIE-style interpretation prompt (Chen et al., ICML 2024), in LLaVA's USER/ASSISTANT
#          format, with the state repeated over five placeholders. Unlike the original SelfIE we
#          inject at the source layer (same-layer patching) and score a closed set of class names.
PROMPTS = {
    "identity": (TARGET_PROMPTS["identity"], " -> {}", None),
    "entity": (TARGET_PROMPTS["entity"], ": {}", None),
    # Placeholder spellings per tokenizer: SentencePiece (LLaVA) "▁_"; byte-level BPE (Qwen) "Ġ_",
    # where the last one merges with the newline into "Ġ_Ċ" (still the fifth placeholder).
    "selfie": ("USER: _ _ _ _ _\nASSISTANT: Sure, I'll summarize your message:", " {}",
               (("▁_",), ("Ġ_", "Ġ_Ċ"), ("_",))),
}


def class_forms(class_names, synonyms):
  """{class: [canonical name, unambiguous synonyms...]}."""
  forms = {c: [c] + (list(SURFACE_FORMS.get(c, [])) if synonyms else []) for c in class_names}
  owners = Counter(f.lower() for fs in forms.values() for f in set(x.lower() for x in fs))
  return {c: [fs[0]] + [f for f in dict.fromkeys(fs[1:])
                        if owners[f.lower()] == 1 and f.lower() != fs[0].lower()]
          for c, fs in forms.items()}


def _scatter_max(values, class_index, n_classes):
  """values [n, M], class_index [M] -> [n, n_classes] max per class (-inf if none)."""
  out = torch.full((values.shape[0], n_classes), float("-inf"), device=values.device)
  return out.scatter_reduce(1, class_index.expand(values.shape[0], -1), values.float(),
                            reduce="amax", include_self=True)


class LogitLens:
  name = "logit_lens"

  def __init__(self, mt, class_names, forms, device, chunk=512):
    self.norm = get_decoder(mt).norm
    self.head = mt.model.lm_head
    self.device, self.chunk = device, chunk
    pairs = [(ci, tid) for ci, c in enumerate(class_names)
             for f in forms[c] for tid in class_token_ids(mt.tokenizer, f)]
    pairs = sorted(set(pairs))
    self.n_classes = len(class_names)
    self.cls = torch.tensor([p[0] for p in pairs], device=device)
    self.ids = torch.tensor([p[1] for p in pairs], device=device)

  @torch.no_grad()
  def scores(self, h, layer=None):
    out = []
    dtype = next(self.head.parameters()).dtype
    for start in range(0, h.shape[0], self.chunk):
      x = h[start:start + self.chunk].to(self.device, dtype)
      logp = self.head(self.norm(x)).float().log_softmax(-1)
      out.append(_scatter_max(logp[:, self.ids], self.cls, self.n_classes).cpu())
    return torch.cat(out).numpy()


def rms_norm(x, weight, eps):
  """LLaMA RMSNorm in float32 (same as the decoder's final norm)."""
  x = x.float()
  return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


class TunedLens(LogitLens):
  """LogitLens on h + A_l h + b_l, with translators (A, b) trained by train_tuned_lens.py."""

  name = "tuned_lens"

  def __init__(self, mt, class_names, forms, device, translators_path, chunk=512):
    super().__init__(mt, class_names, forms, device, chunk)
    saved = torch.load(translators_path, map_location="cpu", weights_only=False)
    self.A = saved["A"].to(device, torch.float32)   # [L, d, d]
    self.b = saved["b"].to(device, torch.float32)   # [L, d]
    self.norm_w = self.norm.weight.detach().to(device, torch.float32)
    self.eps = self.norm.variance_epsilon
    self.head_w = self.head.weight.detach().to(device, torch.float32)

  @torch.no_grad()
  def scores(self, h, layer):
    out = []
    for start in range(0, h.shape[0], self.chunk):
      x = h[start:start + self.chunk].to(self.device, torch.float32)
      x = x + x @ self.A[layer].T + self.b[layer]
      logp = (rms_norm(x, self.norm_w, self.eps) @ self.head_w.T).log_softmax(-1)
      out.append(_scatter_max(logp[:, self.ids], self.cls, self.n_classes).cpu())
    return torch.cat(out).numpy()


class EmbeddingLens:
  """Nearest input embedding: max cosine between the state and a class's first sub-tokens.

  The EmbeddingLens baseline of LatentLens (Krojer et al., 2026), scored closed-set.
  """

  name = "embedding_lens"

  def __init__(self, mt, class_names, forms, device, chunk=2048):
    pairs = sorted({(ci, tid) for ci, c in enumerate(class_names)
                    for f in forms[c] for tid in class_token_ids(mt.tokenizer, f)})
    self.n_classes, self.device, self.chunk = len(class_names), device, chunk
    self.cls = torch.tensor([p[0] for p in pairs], device=device)
    emb = mt.model.get_input_embeddings().weight.detach()
    ids = torch.tensor([p[1] for p in pairs], device=emb.device)
    self.emb = F.normalize(emb[ids].float(), dim=-1).to(device)

  @torch.no_grad()
  def scores(self, h, layer=None):
    out = []
    for start in range(0, h.shape[0], self.chunk):
      q = F.normalize(h[start:start + self.chunk].float(), dim=-1).to(self.device)
      out.append(_scatter_max(q @ self.emb.T, self.cls, self.n_classes).cpu())
    return torch.cat(out).numpy()


class LatentLens:
  name = "latentlens"

  def __init__(self, bank_dir, tokenizer, class_names, forms, device, cache_file=None):
    self.device = device
    self.n_classes = len(class_names)
    layer_files = sorted(
        (int(d.split("_")[1]), os.path.join(bank_dir, d, "embeddings_cache.pt"))
        for d in os.listdir(bank_dir) if d.startswith("layer_"))
    if not layer_files:
      raise FileNotFoundError(f"no layer_*/embeddings_cache.pt under {bank_dir}")
    if cache_file and os.path.exists(cache_file):
      saved = np.load(cache_file)
      cols, cls = saved["cols"], saved["cls"]
    else:
      # Metadata is identical for every bank layer; match class names on the first one.
      first = torch.load(layer_files[0][1], map_location="cpu", weights_only=False)
      cols, cls = _match_bank_entries(first["metadata"], tokenizer, class_names, forms)
      del first
      if cache_file:
        np.savez(cache_file, cols=cols, cls=cls)
    self.cls = torch.as_tensor(cls, device=device)
    dtype = torch.float16 if torch.device(device).type == "cuda" else torch.float32
    # Only bank entries that name some class can contribute a class score; load one layer at a
    # time and keep just those rows (the full bank is ~3 GB per layer).
    self.emb = {}
    for layer, path in layer_files:
      data = torch.load(path, map_location="cpu", weights_only=False)
      rows = data["embeddings"][torch.as_tensor(cols)].float()
      self.emb[layer] = F.normalize(rows, dim=-1).to(device, dtype)
      del data
    per_class = np.bincount(cls, minlength=self.n_classes)
    self.missing = [c for c, k in zip(class_names, per_class) if k == 0]
    if self.missing:
      print(f"  LatentLens: no bank entry names {self.missing} (always ranked last)")

  @torch.no_grad()
  def scores(self, h, layer=None):
    q = F.normalize(h.float(), dim=-1).to(self.device)
    best = None
    for l, emb in self.emb.items():
      s = _scatter_max(q.to(emb.dtype) @ emb.T, self.cls, self.n_classes)
      best = s if best is None else torch.maximum(best, s)
    return best.cpu().numpy()


def _match_bank_entries(meta, tokenizer, class_names, forms):
  """(cols, cls): bank entry j names class c (whole word in its sentence, any form)."""
  lowered = {c: [f.lower() for f in forms[c]] for c in class_names}
  cols, cls = [], []
  ctx_cache = {}
  for j, m in enumerate(meta):
    caption = m["caption"].lower()
    # Cheap pre-filter before the tokenizer-based word check: the sentence must contain a form.
    hits = [ci for ci, c in enumerate(class_names) if any(f in caption for f in lowered[c])]
    if not hits:
      continue
    key = (m["caption"], m["position"])
    if key not in ctx_cache:
      ctx_cache[key] = word_context(tokenizer, m["caption"], m["position"], m["token_id"])
    ctx = ctx_cache[key]
    for ci in hits:
      if word_matches(ctx, forms[class_names[ci]]):
        cols.append(j)
        cls.append(ci)
    if (j + 1) % 50000 == 0:
      print(f"    matched {j + 1}/{len(meta)} bank entries")
  return np.asarray(cols, dtype=np.int64), np.asarray(cls, dtype=np.int64)


class Patchscopes:
  """Closed-set Patchscopes: likelihood of each class-naming continuation after injection."""

  name = "patchscopes"

  def __init__(self, mt, class_names, forms, prompt_key, device, rows_per_batch=256,
               use_cache=True):
    self.mt, self.device = mt, device
    self.decoder = get_decoder(mt)
    tok = mt.tokenizer
    prompt, continuation, placeholder = PROMPTS[prompt_key]
    self.prompt_ids = tok(prompt)["input_ids"]
    if placeholder is None:
      self.pos = [len(self.prompt_ids) - 1]  # the last prompt token ('?' or 'x')
    else:
      # The placeholder's spelling differs per tokenizer (SentencePiece "▁_", byte-level BPE
      # "Ġ_"); use the first candidate that occurs in the tokenized prompt.
      candidates = placeholder if isinstance(placeholder, tuple) else (placeholder,)
      self.pos = []
      for cand in candidates:
        spellings = cand if isinstance(cand, tuple) else (cand,)
        ids = {tok.convert_tokens_to_ids(t) for t in spellings} - {None, tok.unk_token_id}
        self.pos = [i for i, t in enumerate(self.prompt_ids) if t in ids]
        if self.pos:
          break
      if not self.pos:
        raise ValueError(f"placeholder {placeholder!r} not found in the {prompt_key} prompt")
    conts, owners = [], []
    for ci, c in enumerate(class_names):
      for f in forms[c]:
        full = tok(prompt + continuation.format(f))["input_ids"]
        if full[:len(self.prompt_ids)] != self.prompt_ids:
          raise ValueError(f"prompt tokenization changes when followed by {f!r}")
        conts.append(full[len(self.prompt_ids):])
        owners.append(ci)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    width = max(len(c) for c in conts)
    self.cont = torch.full((len(conts), width), pad, dtype=torch.long)
    self.cont_mask = torch.zeros((len(conts), width), dtype=torch.long)
    for i, c in enumerate(conts):
      self.cont[i, :len(c)] = torch.tensor(c)
      self.cont_mask[i, :len(c)] = 1
    self.owner = torch.tensor(owners)
    self.n_seq, self.n_classes = len(conts), len(class_names)
    self.batch = max(1, rows_per_batch // self.n_seq)
    self.use_cache = use_cache
    self.prior = self._seq_logp(None, None)[0]  # [S], prompt alone

  def _hook(self, layer, states):
    if states is None:
      return None

    def hook(module, inp, out):
      hs = out[0] if isinstance(out, tuple) else out
      if hs.shape[1] > max(self.pos):  # the prompt pass, not continuation-only passes
        hs[:, self.pos] = states.to(hs.dtype)[:, None, :]
    return self.decoder.layers[layer].register_forward_hook(hook)

  @torch.no_grad()
  def _seq_logp(self, states, layer):
    """Log-likelihood of every continuation, for each injected state. -> [B, S]."""
    n = 1 if states is None else states.shape[0]
    S, P = self.n_seq, len(self.prompt_ids)
    cont = self.cont.to(self.device)
    mask = self.cont_mask.to(self.device)
    states = None if states is None else states.to(self.device)

    if self.use_cache:
      ids = torch.tensor([self.prompt_ids] * n, device=self.device)
      handle = self._hook(layer, states)
      try:
        out = self.mt.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True,
                            position_ids=text_position_ids(self.mt, 0, P, n, self.device))
      finally:
        if handle is not None:
          handle.remove()
      past = out.past_key_values
      if not hasattr(past, "batch_repeat_interleave"):
        self.use_cache = False
        return self._seq_logp(None if states is None else states.cpu(), layer)
      first = out.logits[:, -1].float().log_softmax(-1).repeat_interleave(S, 0)  # [n*S, V]
      past.batch_repeat_interleave(S)
      c = cont.repeat(n, 1)
      m = mask.repeat(n, 1)
      attn = torch.cat([torch.ones((n * S, P), dtype=torch.long, device=self.device), m], 1)
      logits = self.mt.model(input_ids=c, attention_mask=attn, past_key_values=past,
                             use_cache=False,
                             position_ids=text_position_ids(self.mt, P, c.shape[1], n * S,
                                                            self.device)
                             ).logits.float().log_softmax(-1)  # [n*S, W, V]
      tok_lp = torch.empty(c.shape, device=self.device)
      tok_lp[:, 0] = first.gather(1, c[:, :1]).squeeze(1)
      if c.shape[1] > 1:
        tok_lp[:, 1:] = logits[:, :-1].gather(2, c[:, 1:].unsqueeze(-1)).squeeze(-1)
    else:
      # Fallback without a reusable cache: run prompt + continuation in full.
      seq = torch.cat([torch.tensor([self.prompt_ids], device=self.device).repeat(n * S, 1),
                       cont.repeat(n, 1)], 1)
      m = mask.repeat(n, 1)
      attn = torch.cat([torch.ones((n * S, P), dtype=torch.long, device=self.device), m], 1)
      handle = self._hook(layer, None if states is None else states.repeat_interleave(S, 0))
      try:
        logits = self.mt.model(input_ids=seq, attention_mask=attn,
                               position_ids=text_position_ids(self.mt, 0, seq.shape[1], n * S,
                                                              self.device)).logits.float()
      finally:
        if handle is not None:
          handle.remove()
      lp = logits[:, P - 1:-1].log_softmax(-1)
      c = seq[:, P:]
      tok_lp = lp.gather(2, c.unsqueeze(-1)).squeeze(-1)
    total = (tok_lp * m).sum(1)
    return total.view(n, S).cpu()

  def scores(self, h, layer):
    """Returns (raw, pmi), each [n, n_classes]."""
    raw, pmi = [], []
    for start in range(0, h.shape[0], self.batch):
      s = self._seq_logp(h[start:start + self.batch], layer)
      raw.append(_scatter_max(s, self.owner, self.n_classes))
      pmi.append(_scatter_max(s - self.prior, self.owner, self.n_classes))
    return torch.cat(raw).numpy(), torch.cat(pmi).numpy()


def probe_scores(X, meta, class_names, kind, n_splits, seed, high_coverage=0.5):
  """Grouped cross-validated probe log-probabilities for every row. -> [n, n_classes].

  Trained on object tokens with coverage >= `high_coverage`; every row of a held-out image
  (all bins, pooled, outside, random) is scored by a probe that never saw that image.
  """
  from sklearn.linear_model import LogisticRegression
  from sklearn.neural_network import MLPClassifier
  from sklearn.pipeline import make_pipeline
  from sklearn.preprocessing import StandardScaler

  index = {c: i for i, c in enumerate(class_names)}
  y = meta["class_name"].map(index).to_numpy()
  train_ok = ((meta["kind"] == "object") & (meta["coverage"] >= high_coverage)).to_numpy()
  images = meta["image_id"].unique()
  rng = np.random.RandomState(seed)
  fold_of = dict(zip(rng.permutation(images), np.arange(len(images)) % n_splits))
  fold = meta["image_id"].map(fold_of).to_numpy()

  out = np.full((len(meta), len(class_names)), -np.inf, dtype=np.float32)
  for k in range(n_splits):
    train = train_ok & (fold != k)
    test = fold == k
    if len(np.unique(y[train])) < 2 or not test.any():
      continue
    if kind == "linear":
      clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    else:
      # Early stopping holds out 10% of the training rows; only possible with enough data.
      enough = bool(train.sum() >= 20 * len(np.unique(y[train])))
      clf = make_pipeline(StandardScaler(),
                          MLPClassifier(hidden_layer_sizes=(256,), early_stopping=enough,
                                        max_iter=300, random_state=seed))
    clf.fit(X[train], y[train])
    proba = clf.predict_proba(X[test])
    cols = clf.classes_
    block = np.full((test.sum(), len(class_names)), -np.inf, dtype=np.float32)
    block[:, cols] = np.log(proba + 1e-12)
    out[test] = block
  return out
