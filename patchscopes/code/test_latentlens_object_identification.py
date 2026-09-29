"""CPU smoke test for latentlens_object_identification.py.

Builds a contextual bank with a tiny randomly-initialised LLaVA decoder and checks the parts
that could silently corrupt the comparison: layer indexing (our layer l == HF / LatentLens
layer l+1), prefix de-duplication and the per-token cap, cross-layer search, the word-level
scoring, and the end-to-end query + summary on a fake results directory. If the official
`latentlens` package is importable, search results are also checked against
`latentlens.ContextualIndex.search`.

Needs network access once, to fetch the LLaVA processor/tokenizer config (no weights).

  python test_latentlens_object_identification.py
"""

import os
import sys
import tempfile

import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import latentlens_object_identification as L  # noqa: E402
from test_visual_object_identification import build_tiny_vlm  # noqa: E402

CORPUS = [
    "A small dog ran across the park.",
    "The cat slept on the warm windowsill.",
    "We ate dinner at the dining table last night.",
    "A refrigerator keeps food cold.",
    "A small dog ran across the park.",  # exact duplicate: must add nothing
    "Two dogs and a cat played in the garden.",
]


def capture_block_outputs(decoder, tokenizer, text):
  """Output of every decoder block for one sentence, via forward hooks (pre final-norm)."""
  enc = tokenizer(text, return_tensors="pt")
  captured = {}
  handles = [
      layer.register_forward_hook(
          lambda m, i, o, idx=idx: captured.__setitem__(
              idx, (o[0] if isinstance(o, tuple) else o)[0].detach().clone()))
      for idx, layer in enumerate(decoder.layers)]
  try:
    with torch.no_grad():
      decoder(**enc)
  finally:
    for h in handles:
      h.remove()
  return enc["input_ids"][0].tolist(), captured


def find_position(tokenizer, text, word):
  """Position of the first sub-token of `word` in the tokenised sentence."""
  ids = tokenizer(text)["input_ids"]
  target = tokenizer.encode(" " + word, add_special_tokens=False)[0]
  alt = tokenizer.encode(word, add_special_tokens=False)[0]
  for pos, tok in enumerate(ids):
    if tok in (target, alt):
      return pos
  raise ValueError(f"{word!r} not found in {text!r}")


def test_word_context(tokenizer):
  text = "We ate dinner at the dining table last night."
  pos = find_position(tokenizer, text, "table")
  ids = tokenizer(text)["input_ids"]
  before, word, after = L.word_context(tokenizer, text, pos, ids[pos])
  assert word == "table", (before, word, after)
  assert before[-1] == "dining" and after[0] == "last", (before, after)
  assert L.word_matches((before, word, after), ["dining table"])
  assert L.word_matches((before, word, after), ["table"])
  assert not L.word_matches((before, word, after), ["dinner"])  # neighbour, not the token

  text = "A refrigerator keeps food cold."
  pos = find_position(tokenizer, text, "refrigerator")
  ids = tokenizer(text)["input_ids"]
  _, word, _ = L.word_context(tokenizer, text, pos, ids[pos])
  assert word == "refrigerator", word  # a sub-token expands to the whole word
  assert L.word_context(tokenizer, text, pos, ids[pos] + 1) is None  # token mismatch
  print("word context OK")


def test_bank(decoder, tokenizer):
  bank_layers = [1, 2, 3]  # tiny model has 4 blocks; HF index 4 would be post-norm
  bank = L.build_bank(decoder, tokenizer, CORPUS, bank_layers, max_contexts_per_token=50,
                      batch_size=4)
  meta = bank.layers_data[1]["metadata"]
  n = len(meta)
  assert all(bank.layers_data[l]["embeddings"].shape == (n, 64) for l in bank_layers)
  assert all(m["position"] >= 2 for m in meta)
  # The duplicate sentence contributes nothing: no (caption, position) pair appears twice.
  assert len({(m["caption"], m["position"]) for m in meta}) == n

  # Layer indexing: HF layer i == output of block i-1, for the stored (normalised) vectors.
  text = CORPUS[0]
  ids, blocks = capture_block_outputs(decoder, tokenizer, text)
  row = next(i for i, m in enumerate(meta) if m["caption"] == text and m["position"] == 3)
  for hf_layer in bank_layers:
    stored = bank.layers_data[hf_layer]["embeddings"][row].float()
    expected = F.normalize(blocks[hf_layer - 1][3].float(), dim=0)
    assert torch.allclose(stored, expected, atol=2e-3), hf_layer
  print(f"bank OK ({n} contexts, layer indexing verified)")

  # Per-token cap.
  capped = L.build_bank(decoder, tokenizer, CORPUS, [1], max_contexts_per_token=1,
                        batch_size=4)
  tokens = [m["token_str"] for m in capped.layers_data[1]["metadata"]]
  assert len(tokens) == len(set(tokens)), "cap of 1 context per token violated"

  try:
    L.build_bank(decoder, tokenizer, CORPUS, [4])
    raise AssertionError("post-norm bank layer accepted")
  except ValueError:
    pass
  print("cap + layer validation OK")
  return bank, blocks, ids


def test_search(bank):
  # A stored vector must retrieve itself with cosine ~1, from the right bank layer.
  query = bank.layers_data[2]["embeddings"][5].float() * 7.3  # scale must not matter
  sims, layers, rows = bank.search(query[None], top_k=5)
  assert int(layers[0, 0]) == 2 and int(rows[0, 0]) == 5, (layers, rows)
  assert abs(float(sims[0, 0]) - 1) < 1e-3
  assert (sims[0, :-1] >= sims[0, 1:]).all(), "not sorted across layers"

  with tempfile.TemporaryDirectory() as tmp:
    bank.save(tmp)
    loaded = L.ContextualBank.load(tmp)
    s2, l2, r2 = loaded.search(query[None], top_k=5)
    assert torch.equal(l2, layers) and torch.equal(r2, rows)

  try:
    import latentlens
  except ImportError:
    print("search OK (official latentlens not installed -- parity check skipped)")
    return
  queries = torch.randn(16, 64)
  official = latentlens.ContextualIndex(
      {l: {"embeddings": d["embeddings"].float(), "metadata": d["metadata"]}
       for l, d in bank.layers_data.items()})
  ref = official.search(F.normalize(queries, dim=-1), top_k=5)
  sims, layers, rows = bank.search(queries, top_k=5)
  for i in range(len(queries)):
    for j in range(5):
      m = bank.meta(layers[i, j], rows[i, j])
      assert m["token_str"] == ref[i][j].token_str and m["caption"] == ref[i][j].caption
      assert int(layers[i, j]) == ref[i][j].contextual_layer
      assert abs(float(sims[i, j]) - ref[i][j].similarity) < 1e-3
  print("search OK (identical to official latentlens.ContextualIndex.search)")


def test_end_to_end(mt, bank):
  """Fake results dir: one query equals the bank's 'dog' state, so it must score correct."""
  tokenizer = mt.tokenizer
  decoder = mt.model.model.language_model
  text = CORPUS[0]
  pos = find_position(tokenizer, text, "dog")
  _, blocks = capture_block_outputs(decoder, tokenizer, text)

  torch.manual_seed(0)
  n_rows, n_layers = 8, 4
  states = torch.randn(n_rows, n_layers, 64).half()
  states[0, 1] = blocks[1][pos].half()  # our layer 1 == HF / bank layer 2
  meta = pd.DataFrame({
      "image_id": [0, 0, 1, 1, 2, 2, 3, 3],
      "file_name": ["x"] * n_rows,
      "class_name": ["dog", "dog", "dog", "dog", "cat", "cat", "cat", "cat"],
      "patch_index": list(range(n_rows)),
      "kind": ["object", "outside"] * 4,
  })

  with tempfile.TemporaryDirectory() as tmp:
    results_dir = os.path.join(tmp, "results")
    index_dir = os.path.join(tmp, "index")
    out_dir = os.path.join(tmp, "out")
    os.makedirs(results_dir)
    torch.save({"states": states, "meta": meta}, os.path.join(results_dir, "patch_states.pt"))
    corpus_path = os.path.join(tmp, "corpus.txt")
    with open(corpus_path, "w", encoding="utf-8") as f:
      f.write("\n".join(CORPUS))
    bank.save(os.path.join(index_dir, "bank"))

    L.main(["--results_dir", results_dir, "--index_dir", index_dir, "--out_dir", out_dir,
            "--corpus", corpus_path, "--device", "cpu", "--n_boot", "50"])

    ll = pd.read_csv(os.path.join(out_dir, "latentlens.csv"))
    assert len(ll) == (n_rows + 4) * n_layers  # + one random control per image
    assert set(ll["kind"]) == {"object", "outside", "random"}
    assert (ll["hf_layer"] == ll["layer"] + 1).all()
    hit = ll[(ll["row_idx"] == 0) & (ll["layer"] == 1) & (ll["kind"] == "object")].iloc[0]
    assert hit["nn_word"] == "dog" and hit["nn_caption"] == text, hit.to_dict()
    assert hit["top1_word"] and hit["top1_token"] and hit["top5_word"], hit.to_dict()
    assert hit["nn_similarity"] > 0.999
    for name in ("latentlens_summary.csv", "latentlens_headline.json",
                 "corpus_class_coverage.csv"):
      assert os.path.exists(os.path.join(out_dir, name)), name
    cov = pd.read_csv(os.path.join(out_dir, "corpus_class_coverage.csv"))
    assert cov.set_index("class_name").loc["dog", "sentences_exact"] == 3
  print("end-to-end OK")


def main():
  mt, _ = build_tiny_vlm()
  decoder = mt.model.model.language_model
  test_word_context(mt.tokenizer)
  bank, _, _ = test_bank(decoder, mt.tokenizer)
  test_search(bank)
  test_end_to_end(mt, bank)
  print("\nALL TESTS PASSED")


if __name__ == "__main__":
  main()
