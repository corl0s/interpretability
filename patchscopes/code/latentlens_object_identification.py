r"""LatentLens readout on the cached visual patch states (LLaVA-1.5), scored like the others.

Adds a fourth readout to `visual_object_identification.py`, on the *identical* cached states
(`patch_states.pt`), so all readouts share one ground truth:

  1. Patchscopes  (visual_object_identification.py)
  2. Logit lens   (visual_object_identification.py)
  3. Linear probe (visual_object_identification.py)
  4. LatentLens   (this script) -- Krojer et al., ICML 2026, arXiv 2602.00462.

LatentLens describes a hidden state by its nearest neighbours in a bank of *contextual* text
token representations: the LLM is run over a text corpus, the hidden state of every token (in
context) is stored at a few layers, and a visual token is compared (cosine) against all of
them at once, with the top-k merged across layers. There is no pre-built bank for LLaVA-1.5,
so this script builds one from LLaVA-1.5's own language model -- not base Vicuna, because
LLaVA fine-tunes the LLM, and the bank must live in the same space as the visual states.

Bank construction and search are re-implemented here (~80 lines) rather than imported, because
the PyPI release (0.1.0) and GitHub `main` of `latentlens` have different APIs and the package
requires Python >= 3.10. The algorithm follows GitHub `main` (`latentlens/extract.py`,
`latentlens/index.py`) exactly: positions 0-1 skipped, prefix de-duplication, at most
`max_contexts_per_token` contexts per token string, L2-normalised embeddings, per-layer top-k
followed by a global cross-layer merge. `test_latentlens_object_identification.py` checks the
search against the official `ContextualIndex.search` when the package is installed.

Layer convention. Everywhere in this project, layer `l` is the OUTPUT of decoder block `l`
(pre final-norm). HF `hidden_states[i]` is the output of block `i-1`, and LatentLens uses the HF
index. So our layer `l` == LatentLens layer `l+1`. Both are written to the CSV (`layer`,
`hf_layer`). Bank layers are given in the HF convention and must be < num_layers, because HF's
last entry is post final-norm and would not match the pre-norm visual states.

Scoring (for each of top-1 and top-5 neighbours):
  * token -- the neighbour token is a first sub-token of the class name: the logit-lens
             criterion, so the two readouts are compared like for like.
  * word  -- the whole word containing the neighbour token, in its source sentence, names the
             class (a 1-2 word window on each side for multi-word classes). LatentLens returns
             tokens in context, so "ref" inside "refrigerator" counts; this is the fairer number.
  * word_lenient -- as `word`, with the curated synonyms from analyze_object_identification.py.
Controls: outside-object patches and random vectors (one per image), as for Patchscopes.
Headline numbers use the same 'clean' classes as analyze_object_identification.py (zero control
false-positives for Patchscopes), read from `class_control_fp.csv`.

Usage (SCC, one GPU):

  # 0. does the corpus cover the COCO classes? (CPU, seconds)
  python latentlens_object_identification.py --stage coverage

  # 1. smoke test: small bank, a few layers (~1 h)
  python latentlens_object_identification.py --corpus_limit 10000 \
      --index_dir ./results/latentlens_index_smoke --out_dir ./results/latentlens_smoke

  # 2. full run: whole corpus (~117k sentences), 8 bank layers
  python latentlens_object_identification.py

Stages are checkpointed: the bank is reused unless --overwrite is passed.
"""

import argparse
import json
import os
import random
import re
import sys
import urllib.request
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyze_object_identification import (  # noqa: E402
    SURFACE_FORMS,
    bootstrap_by_image,
)
from visual_object_identification import class_token_ids, mentions  # noqa: E402

CORPUS_URL = "https://raw.githubusercontent.com/McGill-NLP/latentlens/main/concepts.txt"
# LatentLens's default grid for 32-layer LLMs (HF hidden_states index; see module docstring).
DEFAULT_BANK_LAYERS = [1, 2, 4, 8, 16, 24, 30, 31]


# --------------------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------------------


def load_corpus(path, index_dir, limit=0, seed=0):
  """One sentence per line. Downloads LatentLens's bundled concepts.txt if no path given."""
  if path is None:
    path = os.path.join(index_dir, "concepts.txt")
    if not os.path.exists(path):
      os.makedirs(index_dir, exist_ok=True)
      print(f"Downloading LatentLens corpus to {path} ...")
      urllib.request.urlretrieve(CORPUS_URL, path)
  with open(path, encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]
  if limit and limit < len(texts):
    texts = random.Random(seed).sample(texts, limit)
  return texts


def corpus_coverage(texts, class_names):
  """Number of corpus sentences mentioning each class name (exact, and with synonyms)."""
  rows = []
  for name in sorted(class_names):
    forms = [name] + SURFACE_FORMS.get(name, [])
    rows.append({
        "class_name": name,
        "sentences_exact": sum(mentions(t, name) for t in texts),
        "sentences_lenient": sum(any(mentions(t, f) for f in forms) for t in texts),
    })
  return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Contextual bank (LatentLens algorithm)
# --------------------------------------------------------------------------------------


class ContextualBank:
  """Per-layer L2-normalised contextual embeddings + metadata, searched LatentLens-style."""

  def __init__(self, layers_data):
    # {hf_layer: {"embeddings": Tensor[N, D] (normalised), "metadata": list[dict]}}
    self.layers_data = layers_data

  @property
  def layers(self):
    return sorted(self.layers_data)

  def __repr__(self):
    sizes = {l: self.layers_data[l]["embeddings"].shape[0] for l in self.layers}
    return f"ContextualBank(layers={self.layers}, sizes={sizes})"

  def save(self, path):
    os.makedirs(path, exist_ok=True)
    for layer, data in self.layers_data.items():
      layer_dir = os.path.join(path, f"layer_{layer}")
      os.makedirs(layer_dir, exist_ok=True)
      # Same file layout as latentlens.ContextualIndex, so either can load the other's bank.
      torch.save({"embeddings": data["embeddings"].cpu().half(),
                  "metadata": data["metadata"]},
                 os.path.join(layer_dir, "embeddings_cache.pt"))
    with open(os.path.join(path, "metadata.json"), "w", encoding="utf-8") as f:
      json.dump({"layers": self.layers}, f)

  @classmethod
  def load(cls, path):
    layers_data = {}
    for name in sorted(os.listdir(path)):
      cache = os.path.join(path, name, "embeddings_cache.pt")
      if name.startswith("layer_") and os.path.exists(cache):
        data = torch.load(cache, map_location="cpu", weights_only=False)
        layers_data[int(name.split("_")[1])] = {
            "embeddings": F.normalize(data["embeddings"].float(), dim=-1).half(),
            "metadata": data["metadata"],
        }
    if not layers_data:
      raise FileNotFoundError(f"no layer_*/embeddings_cache.pt under {path}")
    return cls(layers_data)

  def search(self, queries, top_k=5, device="cpu", chunk=4096):
    """Top-k neighbours per query, merged across all bank layers.

    queries: [Q, D]. Returns (sims [Q, k], bank_layer [Q, k], row [Q, k]) as CPU tensors,
    sorted by descending cosine similarity. Each bank layer is moved to `device` once.
    """
    queries = F.normalize(queries.float(), dim=-1)
    dtype = torch.float16 if torch.device(device).type == "cuda" else torch.float32
    all_vals, all_rows, all_layers = [], [], []
    for layer in self.layers:
      emb = self.layers_data[layer]["embeddings"].to(device, dtype)
      k = min(top_k, emb.shape[0])
      vals, rows = [], []
      for start in range(0, queries.shape[0], chunk):
        q = queries[start:start + chunk].to(device, dtype)
        v, r = torch.topk(q @ emb.T, k=k, dim=-1)
        vals.append(v.float().cpu())
        rows.append(r.cpu())
      all_vals.append(torch.cat(vals))
      all_rows.append(torch.cat(rows))
      all_layers.append(torch.full_like(all_rows[-1], layer))
      del emb
    vals = torch.cat(all_vals, dim=1)      # [Q, n_layers * k]
    rows = torch.cat(all_rows, dim=1)
    layers = torch.cat(all_layers, dim=1)
    merge_k = min(top_k, vals.shape[1])
    top_vals, pos = torch.topk(vals, k=merge_k, dim=-1)
    return top_vals, layers.gather(1, pos), rows.gather(1, pos)

  def meta(self, layer, row):
    return self.layers_data[int(layer)]["metadata"][int(row)]


def build_bank(decoder, tokenizer, texts, bank_layers, max_contexts_per_token=50,
               batch_size=32, max_length=512):
  """Run the LLM over the corpus and store contextual token states (LatentLens extract.py).

  `decoder` is LLaVA's bare language model; hidden_states[i] is the output of block i-1.
  """
  n_layers = decoder.config.num_hidden_layers
  bad = [l for l in bank_layers if not 1 <= l < n_layers]
  if bad:
    raise ValueError(f"bank layers {bad} out of range: use 1..{n_layers - 1} (HF index; "
                     f"{n_layers} is post final-norm and does not match the visual states)")
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
  # The position loop below assumes real tokens occupy 0..valid_len-1.
  tokenizer.padding_side = "right"
  device = next(decoder.parameters()).device

  embeddings = defaultdict(list)
  metadata = []
  seen_prefixes = set()
  token_counts = defaultdict(int)

  for start in range(0, len(texts), batch_size):
    batch = texts[start:start + batch_size]
    enc = tokenizer(batch, return_tensors="pt", truncation=True, max_length=max_length,
                    padding=True)
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    with torch.no_grad():
      hidden = decoder(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=True, use_cache=False).hidden_states
    for layer in bank_layers:
      if not torch.isfinite(hidden[layer]).all():
        raise RuntimeError(f"non-finite hidden states at HF layer {layer}; try bfloat16")

    keep_sent, keep_pos = [], []
    for s in range(input_ids.shape[0]):
      ids = input_ids[s].tolist()
      valid_len = int(attention_mask[s].sum().item())
      for pos in range(2, valid_len):  # skip BOS and position 1, as LatentLens does
        prefix_hash = hash(tuple(ids[:pos + 1]))
        if prefix_hash in seen_prefixes:
          continue
        token_id = ids[pos]
        token_str = tokenizer.decode([token_id])
        if token_counts[token_str] >= max_contexts_per_token:
          continue
        seen_prefixes.add(prefix_hash)
        token_counts[token_str] += 1
        keep_sent.append(s)
        keep_pos.append(pos)
        metadata.append({"token_str": token_str, "token_id": token_id,
                         "caption": batch[s], "position": pos})
    if keep_sent:
      si = torch.tensor(keep_sent, device=device)
      pi = torch.tensor(keep_pos, device=device)
      for layer in bank_layers:
        embeddings[layer].append(
            F.normalize(hidden[layer][si, pi].float(), dim=-1).half().cpu())

    done = min(start + batch_size, len(texts))
    if (start // batch_size) % 100 == 0 or done == len(texts):
      print(f"  bank: {done}/{len(texts)} sentences, {len(metadata)} contexts")
    del hidden

  return ContextualBank({layer: {"embeddings": torch.cat(embeddings[layer]),
                                 "metadata": metadata}
                         for layer in bank_layers})


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------


_WORD = r"[A-Za-z0-9'\-]"


def word_context(tokenizer, caption, position, token_id, width=2):
  """(words_before, word, words_after) around the token at `position` in its sentence.

  `word` is the complete word the neighbour token belongs to ("ref" -> "refrigerator").
  Returns None if re-tokenising the sentence does not reproduce the stored token.
  """
  ids = tokenizer(caption, truncation=True, max_length=512)["input_ids"]
  if position >= len(ids) or ids[position] != token_id:
    return None
  full = tokenizer.decode(ids[1:], skip_special_tokens=True)
  prefix = tokenizer.decode(ids[1:position + 1], skip_special_tokens=True)
  if not full.startswith(prefix):
    return None
  rest = full[len(prefix):]
  head = re.search(_WORD + r"*$", prefix).group(0)
  tail = re.match(_WORD + r"*", rest).group(0)
  before = re.findall(_WORD + "+", prefix[:len(prefix) - len(head)])[-width:]
  after = re.findall(_WORD + "+", rest[len(tail):])[:width]
  return before, head + tail, after


def word_matches(context, forms):
  """True if some form matches a window of consecutive words that includes the token's word."""
  if context is None:
    return False
  before, word, after = context
  if not word:
    return False
  words = before + [word] + after
  centre = len(before)
  for form in forms:
    n = len(form.split())
    for start in range(max(0, centre - n + 1), centre + 1):
      window = words[start:start + n]
      if len(window) == n and mentions(" ".join(window), form):
        return True
  return False


# --------------------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------------------


def query_rows(meta_df):
  """Object + outside patches from the cache, plus one random-direction control per image."""
  rows = []
  for row_idx, meta in meta_df.iterrows():
    rows.append({"row_idx": row_idx, "kind": meta["kind"], "class_name": meta["class_name"],
                 "image_id": meta["image_id"], "patch_index": meta["patch_index"]})
  first = meta_df[meta_df["kind"] == "object"].groupby("image_id").head(1)
  for row_idx, meta in first.iterrows():
    rows.append({"row_idx": row_idx, "kind": "random", "class_name": meta["class_name"],
                 "image_id": meta["image_id"], "patch_index": meta["patch_index"]})
  return pd.DataFrame(rows)


def run_query(bank, states, meta_df, tokenizer, layers, args):
  """Search every (query row, layer) and score top-1 / top-5 neighbours."""
  rows = query_rows(meta_df)
  gen = torch.Generator().manual_seed(args.seed)
  hidden = states.shape[-1]
  random_vecs = torch.randn(len(rows), hidden, generator=gen)  # cosine search: norm irrelevant

  class_ids = {c: class_token_ids(tokenizer, c) for c in rows["class_name"].unique()}
  forms = {c: [c] + SURFACE_FORMS.get(c, []) for c in rows["class_name"].unique()}
  context_cache = {}

  def context(layer, row):
    key = (int(layer), int(row))
    if key not in context_cache:
      m = bank.meta(layer, row)
      context_cache[key] = word_context(tokenizer, m["caption"], m["position"], m["token_id"])
    return context_cache[key]

  records = []
  for layer in layers:
    q = torch.stack([
        random_vecs[i] if r["kind"] == "random" else states[r["row_idx"], layer].float()
        for i, (_, r) in enumerate(rows.iterrows())])
    sims, bank_layers, bank_rows = bank.search(q, top_k=args.top_k, device=args.device)
    for i, (_, r) in enumerate(rows.iterrows()):
      c = r["class_name"]
      hits = {"token": [], "word": [], "word_lenient": []}
      tokens = []
      # Neighbours are sorted by similarity, so index 0 is the top-1.
      for j in range(sims.shape[1]):
        m = bank.meta(bank_layers[i, j], bank_rows[i, j])
        tokens.append(m["token_str"])
        ctx = context(bank_layers[i, j], bank_rows[i, j])
        hits["token"].append(m["token_id"] in class_ids[c])
        hits["word"].append(word_matches(ctx, [c]))
        hits["word_lenient"].append(word_matches(ctx, forms[c]))
      top = bank.meta(bank_layers[i, 0], bank_rows[i, 0])
      top_ctx = context(bank_layers[i, 0], bank_rows[i, 0])
      records.append({
          **r.to_dict(), "layer": layer, "hf_layer": layer + 1,
          "nn_token": top["token_str"],
          "nn_word": top_ctx[1] if top_ctx else "",
          "nn_caption": top["caption"],
          "nn_bank_layer": int(bank_layers[i, 0]),
          "nn_similarity": float(sims[i, 0]),
          "nn_topk_tokens": "|".join(tokens),
          **{f"top1_{name}": bool(h[0]) for name, h in hits.items()},
          **{f"top{args.top_k}_{name}": any(h) for name, h in hits.items()},
      })
    print(f"  layer {layer}: object top-1 word acc = "
          f"{np.mean([x['top1_word'] for x in records[-len(rows):] if x['kind'] == 'object']):.3f}")
  return pd.DataFrame(records)


def summarize(ll, args):
  """Image-clustered CIs per layer, restricted to the same clean classes as the rescore."""
  fp_path = os.path.join(args.results_dir, "class_control_fp.csv")
  if os.path.exists(fp_path):
    fp = pd.read_csv(fp_path, index_col=0)["control_fp_exact"]
    clean = set(fp[fp == 0].index)
    print(f"clean classes (from {fp_path}): {len(clean)}")
  else:
    clean = set(ll["class_name"].unique())
    print(f"{fp_path} not found -- run analyze_object_identification.py first; "
          f"using all {len(clean)} classes")
  sub = ll[ll["class_name"].isin(clean)]

  frames = []
  for kind in ("object", "outside", "random"):
    kind_df = sub[sub["kind"] == kind]
    if not len(kind_df):
      continue
    for col in [c for c in ll.columns if re.match(r"top\d+_(token|word|word_lenient)$", c)]:
      frame = bootstrap_by_image(kind_df.rename(columns={col: "score"}), "score",
                                 args.n_boot, args.seed)
      frame["condition"] = f"latentlens_{kind}_{col}"
      frames.append(frame)
  summary = pd.concat(frames, ignore_index=True)
  summary.to_csv(os.path.join(args.out_dir, "latentlens_summary.csv"), index=False)

  def curve(cond):
    return summary[summary.condition == cond].set_index("layer")

  obj = curve("latentlens_object_top1_word")
  headline = {
      "n_images": int(sub["image_id"].nunique()),
      "n_clean_classes": len(clean),
      "best_layer": int(obj["mean"].idxmax()),
      "at_layer": {},
  }
  for layer in sorted({2, 29, headline["best_layer"]} & set(obj.index)):
    headline["at_layer"][str(layer)] = {
        cond: {"mean": round(float(curve(cond).loc[layer, "mean"]), 4),
               "ci": [round(float(curve(cond).loc[layer, "ci_lo"]), 4),
                      round(float(curve(cond).loc[layer, "ci_hi"]), 4)]}
        for cond in ("latentlens_object_top1_word", "latentlens_object_top1_token",
                     "latentlens_object_top1_word_lenient", "latentlens_outside_top1_word",
                     "latentlens_random_top1_word")
        if cond in set(summary.condition)}
  with open(os.path.join(args.out_dir, "latentlens_headline.json"), "w",
            encoding="utf-8") as f:
    json.dump(headline, f, indent=2)
  print(json.dumps(headline, indent=2))
  plot(summary, args)
  return summary, headline


def plot(summary, args):
  """All readouts on one figure: LatentLens (here) + Patchscopes / logit lens / probe."""
  try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
  except ImportError:
    return
  fig, ax = plt.subplots(figsize=(10, 5.5))

  def draw(df, label, colour, style):
    ax.plot(df["layer"], df["mean"], style, color=colour, marker="o", ms=3, label=label)
    if "ci_lo" in df:
      ax.fill_between(df["layer"], df["ci_lo"], df["ci_hi"], color=colour, alpha=0.15)

  for cond, label, colour, style in (
      ("latentlens_object_top1_word", "LatentLens top-1 (word)", "C2", "-"),
      ("latentlens_object_top1_token", "LatentLens top-1 (token)", "C2", "--"),
      ("latentlens_outside_top1_word", "LatentLens outside control", "C7", ":"),
      ("latentlens_random_top1_word", "LatentLens random control", "C8", ":")):
    draw(summary[summary.condition == cond], label, colour, style)

  boot = os.path.join(args.results_dir, "bootstrap_by_layer.csv")
  if os.path.exists(boot):
    b = pd.read_csv(boot)
    b = b[b["subset"] == "clean"]
    draw(b[b.condition == "ps_object_exact"], "Patchscopes (entity, exact)", "C0", "-")
    draw(b[b.condition == "logit_lens_top1"], "logit lens top-1", "C1", "-")
  probe = os.path.join(args.results_dir, "linear_probe.csv")
  if os.path.exists(probe):
    p = pd.read_csv(probe).rename(columns={"probe_accuracy": "mean"})
    draw(p, "linear probe (supervised, own class subset)", "C3", "--")

  ax.set_xlabel("LLM backbone layer (output of block l)")
  ax.set_ylabel("object identification accuracy")
  ax.set_title("Readouts of visual patch tokens, LLaVA-1.5-7B on COCO\n"
               "(clean classes, image-clustered 95% CI)")
  ax.legend(fontsize=8)
  ax.grid(alpha=0.3)
  fig.tight_layout()
  path = os.path.join(args.out_dir, "readout_comparison_latentlens.png")
  fig.savefig(path, dpi=150)
  print(f"wrote {path}")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def parse_args(argv=None):
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--stage", default="all", choices=["all", "coverage", "build", "query"])
  p.add_argument("--model_name", default="llava-hf/llava-1.5-7b-hf")
  p.add_argument("--results_dir", default="./results/object_identification",
                 help="output of visual_object_identification.py (patch_states.pt etc.)")
  p.add_argument("--out_dir", default="./results/latentlens")
  p.add_argument("--index_dir", default="./results/latentlens_index")
  p.add_argument("--corpus", default=None,
                 help="one sentence per line; default: LatentLens concepts.txt (downloaded)")
  p.add_argument("--corpus_limit", type=int, default=0, help="random subset size; 0 = all")
  p.add_argument("--bank_layers", default=",".join(map(str, DEFAULT_BANK_LAYERS)),
                 help="HF hidden_states indices (output of block i-1), comma-separated")
  p.add_argument("--layers", default="all",
                 help="visual-state layers to query (our convention), e.g. '0,2,8,29'")
  p.add_argument("--max_contexts_per_token", type=int, default=50)
  p.add_argument("--batch_size", type=int, default=32)
  p.add_argument("--top_k", type=int, default=5)
  p.add_argument("--n_boot", type=int, default=2000)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  p.add_argument("--overwrite", action="store_true")
  return p.parse_args(argv)


def main(argv=None):
  args = parse_args(argv)
  os.makedirs(args.out_dir, exist_ok=True)

  cache = torch.load(os.path.join(args.results_dir, "patch_states.pt"), weights_only=False)
  states, meta_df = cache["states"], cache["meta"]
  n_layers = states.shape[1]
  layers = list(range(n_layers)) if args.layers == "all" else \
      [int(x) for x in args.layers.split(",")]
  print(f"cached states: {tuple(states.shape)}, "
        f"{meta_df['image_id'].nunique()} images, {meta_df['class_name'].nunique()} classes")

  texts = load_corpus(args.corpus, args.index_dir, args.corpus_limit, args.seed)
  print(f"corpus: {len(texts)} sentences")
  coverage = corpus_coverage(texts, meta_df["class_name"].unique())
  coverage.to_csv(os.path.join(args.out_dir, "corpus_class_coverage.csv"), index=False)
  missing = coverage[coverage["sentences_exact"] == 0]["class_name"].tolist()
  print(f"classes never mentioned in the corpus: {missing or 'none'}")
  print(f"fewest mentions:\n{coverage.nsmallest(5, 'sentences_exact').to_string(index=False)}")
  if args.stage == "coverage":
    return

  bank_path = os.path.join(args.index_dir, "bank")
  if os.path.exists(os.path.join(bank_path, "metadata.json")) and not args.overwrite:
    print(f"Loading bank from {bank_path}")
    bank = ContextualBank.load(bank_path)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
  else:
    from general_utils import ModelAndTokenizer
    print(f"Loading {args.model_name} ...")
    mt = ModelAndTokenizer(args.model_name, torch_dtype=torch.float16, device=args.device)
    tokenizer = mt.tokenizer
    bank_layers = [int(x) for x in args.bank_layers.split(",")]
    print(f"Building bank at HF layers {bank_layers} "
          f"(= outputs of blocks {[l - 1 for l in bank_layers]}) ...")
    bank = build_bank(mt.model.model.language_model, tokenizer, texts, bank_layers,
                      args.max_contexts_per_token, args.batch_size)
    bank.save(bank_path)
    del mt
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
  print(bank)
  if args.stage == "build":
    return

  print("Querying cached visual states ...")
  ll = run_query(bank, states, meta_df, tokenizer, layers, args)
  ll.to_csv(os.path.join(args.out_dir, "latentlens.csv"), index=False)
  summarize(ll, args)
  print(f"\nAll outputs in {args.out_dir}")


if __name__ == "__main__":
  main()
