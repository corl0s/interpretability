"""CPU tests for the COCO-VTR Milestone 1 pipeline (vtr/ + run_vtr.py).

Uses a tiny randomly-initialised LLaVA and synthetic COCO images, so accuracies are meaningless;
what is checked is that each piece does what the analysis relies on:
  * class forms drop synonyms shared by several classes
  * pessimistic ranking (ties and unscorable classes cannot help a readout)
  * stratified token sampling by overlap bin; data selection
  * caching: pooled and norm-matched random rows
  * Patchscopes: the fast cached-prefix scoring equals scoring every sequence in full,
    and the injection changes the scores
  * LatentLens: a query equal to a bank entry naming a class scores that class highest
  * probes, evaluation and plots run end to end

Needs network access once, to fetch the LLaVA processor/tokenizer config (no weights).

  python test_vtr.py
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import latentlens_object_identification as LL  # noqa: E402
import run_vtr  # noqa: E402
import train_tuned_lens as T  # noqa: E402
from test_visual_object_identification import build_tiny_vlm, make_fake_coco  # noqa: E402
from vtr import evaluate as E  # noqa: E402
from vtr.cache import cache_states  # noqa: E402
from vtr.data import select_object_samples  # noqa: E402
from vtr.geometry import Llava15Geometry, sample_tokens  # noqa: E402
from visual_object_identification import class_token_ids  # noqa: E402
from vtr.readouts import (EmbeddingLens, LatentLens, LogitLens, Patchscopes,  # noqa: E402
                          TunedLens, class_forms, probe_scores)


def test_forms_and_ranks():
  forms = class_forms(["surfboard", "skateboard", "dining table", "dog"], synonyms=True)
  assert forms["surfboard"] == ["surfboard"], forms["surfboard"]  # "board" shared -> dropped
  assert forms["dining table"] == ["dining table", "table"]
  assert forms["dog"][0] == "dog" and "puppy" in forms["dog"]
  assert class_forms(["dog"], synonyms=False) == {"dog": ["dog"]}

  s = np.array([[0.9, 0.1, 0.0],     # true 0 -> rank 1
                [0.5, 0.5, 0.0],     # tie with true 1 -> pessimistic rank 2
                [-np.inf, 0.0, 0.1]])  # true 0 unscorable -> last
  assert list(E.ranks(s, np.array([0, 1, 0]))) == [1, 2, 3]
  print("forms + ranks OK")


def test_fair_subset():
  names = ["dog", "cat", "baseball bat", "baseball glove", "toaster"]
  rng = np.random.RandomState(0)
  lens = rng.randn(2, 6, 5).astype(np.float32)
  lens[..., 3] = lens[..., 2]                    # logit lens cannot tell bat from glove
  latent = rng.randn(2, 6, 5).astype(np.float32)
  latent[..., 4] = -np.inf                       # LatentLens cannot score toaster
  ps = rng.randn(2, 6, 5).astype(np.float32)
  ps[:, 3:] = np.nan                             # Patchscopes ran on a subset of rows
  scores = {"logit_lens": ([0, 2], lens), "latentlens": ([0, 2], latent),
            "patchscopes_pmi": ([0, 2], ps)}
  keep, dropped = E.fair_classes(scores, ["logit_lens", "latentlens", "patchscopes_pmi"], names)
  assert keep == [0, 1], (keep, dropped)
  assert set(dropped) == {"baseball bat", "baseball glove", "toaster"}
  assert "tied" in dropped["baseball bat"] and "never scored" in dropped["toaster"]

  meta = pd.DataFrame({"class_name": ["dog", "cat", "toaster", "dog", "cat", "baseball bat"],
                       "image_id": [1, 2, 3, 4, 5, 6]})
  sub, sub_meta, sub_names = E.subset(scores, meta, names, keep)
  assert sub_names == ["dog", "cat"] and list(sub_meta["image_id"]) == [1, 2, 4, 5]
  assert sub["latentlens"][1].shape == (2, 4, 2)

  summary = pd.DataFrame({
      "readout": "r", "metric": "top1", "condition": "object_high",
      "layer": [0, 2, 4, 6], "mean": [0.05, 0.1, 0.3, 0.4], "ci_lo": [0.01, 0.05, 0.2, 0.3],
      "ci_hi": [0.1, 0.15, 0.4, 0.5]})
  head = E.headline(summary, n_candidates=2).iloc[0]
  assert head["best_layer"] == 6 and head["half_peak_layer"] == 4 and head["chance"] == 0.5
  print("fair subset + headline OK")


def test_sampling():
  cov = np.zeros(576)
  cov[:3], cov[3:6], cov[6:9], cov[9:20] = 0.1, 0.4, 0.6, 1.0
  picks = sample_tokens(cov, per_bin=2, n_outside=2, rng=np.random.RandomState(0))
  bins = [b for _, b in picks]
  assert bins.count("0-25") == bins.count("25-50") == bins.count("50-75") == 2
  assert bins.count("75-100") == 2 and bins.count("outside") == 2
  assert all(cov[t] == 0 for t, b in picks if b == "outside")
  print("token sampling OK")


def test_patchscopes(mt, class_names, states):
  forms = class_forms(class_names, synonyms=False)
  fast = Patchscopes(mt, class_names, forms, "identity", "cpu", rows_per_batch=8)
  slow = Patchscopes(mt, class_names, forms, "identity", "cpu", rows_per_batch=8,
                     use_cache=False)
  h = states[:3, 2].float()
  raw_fast, pmi_fast = fast.scores(h, 2)
  raw_slow, pmi_slow = slow.scores(h, 2)
  assert np.allclose(raw_fast, raw_slow, atol=1e-3), (raw_fast, raw_slow)
  assert np.allclose(fast.prior.numpy(), slow.prior.numpy(), atol=1e-3)
  assert np.allclose(pmi_fast, raw_fast - fast.prior.numpy()[None, :len(class_names)],
                     atol=1e-3)
  assert not np.allclose(raw_fast[0], fast.prior.numpy()[:len(class_names)], atol=1e-4), \
      "injection had no effect"

  # SelfIE-style prompt: five injection positions, same equivalence.
  sf = Patchscopes(mt, class_names, forms, "selfie", "cpu", rows_per_batch=8)
  ss = Patchscopes(mt, class_names, forms, "selfie", "cpu", rows_per_batch=8, use_cache=False)
  assert len(sf.pos) == 5, sf.pos
  r1, _ = sf.scores(h, 2)
  r2, _ = ss.scores(h, 2)
  assert np.allclose(r1, r2, atol=1e-3), (r1, r2)
  assert not np.allclose(r1[0], sf.prior.numpy()[:len(class_names)], atol=1e-4)
  print("patchscopes + selfie OK (cached scoring == full-sequence scoring)")


def test_lenses(mt, class_names, states, tmp):
  forms = class_forms(class_names, synonyms=False)
  # Embedding lens: the input embedding of a class's first sub-token scores that class highest.
  emb_lens = EmbeddingLens(mt, class_names, forms, "cpu")
  dog_id = sorted(class_token_ids(mt.tokenizer, "dog"))[0]
  q = mt.model.get_input_embeddings().weight[dog_id:dog_id + 1].detach().float() * 5
  s = emb_lens.scores(q)
  assert class_names[int(s.argmax())] == "dog" and s[0].max() > 0.999, s

  # Tuned lens: training lowers held-out KL; zero translators reproduce the logit lens.
  corpus = os.path.join(tmp, "corpus.txt")
  with open(corpus, "w", encoding="utf-8") as f:
    f.write("\n".join(f"Sentence number {i} is about a dog, a cat and a bottle." for i in range(60)))
  out = os.path.join(tmp, "tl", "lens.pt")
  T.main(["--corpus", corpus, "--out", out, "--steps", "30", "--batch_size", "4",
          "--tokens_per_step", "64", "--lr", "1e-2", "--warmup", "1", "--log_every", "10",
          "--eval_fraction", "0.2", "--device", "cpu"], mt=mt)
  saved = torch.load(out, weights_only=False)
  kl_lens, kl_tuned = np.array(saved["kl_logit_lens"]), np.array(saved["kl_tuned_lens"])
  assert kl_tuned[:-1].mean() < kl_lens[:-1].mean(), (kl_lens, kl_tuned)
  assert saved["A"].shape == (4, 64, 64)

  zero = os.path.join(tmp, "tl", "zero.pt")
  torch.save({"A": torch.zeros(4, 64, 64), "b": torch.zeros(4, 64)}, zero)
  h = states[:, 1].float()
  assert np.allclose(TunedLens(mt, class_names, forms, "cpu", zero).scores(h, 1),
                     LogitLens(mt, class_names, forms, "cpu").scores(h), atol=1e-4)
  print(f"embedding lens + tuned lens OK (mean held-out KL {kl_lens[:-1].mean():.3f} -> "
        f"{kl_tuned[:-1].mean():.3f})")
  return out


def test_latentlens(mt, class_names, tmp):
  decoder = mt.model.model.language_model
  corpus = ["A small dog ran across the park.", "The cat slept on the warm windowsill.",
            "She opened a bottle of water.", "Two dogs and a cat played outside."]
  bank = LL.build_bank(decoder, mt.tokenizer, corpus, [1, 2, 3], batch_size=2)
  bank_dir = os.path.join(tmp, "bank")
  bank.save(bank_dir)
  forms = class_forms(class_names, synonyms=False)
  ro = LatentLens(bank_dir, mt.tokenizer, class_names, forms, "cpu",
                  cache_file=os.path.join(tmp, "entries.npz"))
  meta = bank.layers_data[1]["metadata"]
  j = next(i for i, m in enumerate(meta)
           if m["caption"] == corpus[0] and m["token_str"].strip() == "dog")
  q = bank.layers_data[2]["embeddings"][j:j + 1].float() * 3.0
  s = ro.scores(q)
  assert s.shape == (1, len(class_names))
  assert class_names[int(s.argmax())] == "dog" and s[0].max() > 0.99, s
  # Cached entry matches are reused on the second construction.
  ro2 = LatentLens(bank_dir, mt.tokenizer, class_names, forms, "cpu",
                   cache_file=os.path.join(tmp, "entries.npz"))
  assert np.allclose(ro2.scores(q), s)
  print("latentlens OK")
  return bank_dir


def main():
  test_forms_and_ranks()
  test_fair_subset()
  test_sampling()

  with tempfile.TemporaryDirectory() as tmp:
    coco_dir = os.path.join(tmp, "coco")
    ann_path = make_fake_coco(coco_dir)
    geometry = Llava15Geometry()
    samples, class_names = select_object_samples(ann_path, geometry, n_images=4, per_class=3,
                                                 min_area=20000, max_area=None, min_tokens=4,
                                                 seed=0)
    assert len(samples) == 4 and class_names == ["dog", "cat", "bottle"]
    assert all(len(s["coverage"]) == 576 for s in samples)
    print("data selection OK")

    mt, processor = build_tiny_vlm()
    states, meta = cache_states(mt, processor, samples, coco_dir, per_bin=2, n_outside=2,
                                seed=0)
    assert states.shape[1:] == (4, 64)
    kinds = meta["kind"].value_counts().to_dict()
    assert kinds["pooled_mean"] == kinds["pooled_max"] == kinds["random"] == 4
    pm = states[(meta["kind"] == "pooled_mean").to_numpy()].float()
    rnd = states[(meta["kind"] == "random").to_numpy()].float()
    assert torch.allclose(pm.norm(dim=-1), rnd.norm(dim=-1), rtol=1e-2)
    print(f"cache OK {tuple(states.shape)} {kinds}")

    test_patchscopes(mt, class_names, states)
    tuned_path = test_lenses(mt, class_names, states, tmp)
    bank_dir = test_latentlens(mt, class_names, tmp)

    lens = LogitLens(mt, class_names, class_forms(class_names, False), "cpu")
    s = lens.scores(states[:, 1].float())
    assert s.shape == (len(meta), 3) and np.isfinite(s).all()

    probe = probe_scores(states[:, 1].float().numpy(), meta, class_names, "linear", 2, 0)
    assert probe.shape == (len(meta), 3)
    print("logit lens + probe OK")

    # run_vtr's readout stage and evaluation, end to end on the tiny model.
    out_dir = os.path.join(tmp, "out")
    os.makedirs(out_dir)
    args = run_vtr.parse_args(["--out_dir", out_dir, "--device", "cpu",
                               "--latentlens_bank", bank_dir, "--n_boot", "50",
                               "--tuned_lens", tuned_path,
                               "--readouts", "logit_lens", "tuned_lens", "embedding_lens",
                               "latentlens", "patchscopes", "selfie", "probe",
                               "--ps_rows_per_batch", "16", "--ps_max_images", "2",
                               "--probe_splits", "2"])
    layers = [0, 1, 2, 3]
    run_vtr.run_readouts(args, mt, states, meta, class_names, layers)
    names = sorted(f[:-4] for f in os.listdir(os.path.join(out_dir, "scores"))
                   if f.endswith(".npz") and not f.endswith("_entries.npz"))
    assert names == ["embedding_lens", "latentlens", "latentlens_syn", "logit_lens",
                     "logit_lens_syn", "patchscopes_pmi", "patchscopes_raw", "probe_linear",
                     "probe_mlp", "selfie_pmi", "selfie_raw", "tuned_lens"], names
    scores = {n: run_vtr.load_scores(os.path.join(out_dir, "scores", f"{n}.npz"))
              for n in names}
    summary = E.evaluate(scores, meta, class_names, 50, 0)
    ps_rows = summary[(summary.readout == "patchscopes_pmi")
                      & (summary.condition == "object_high")]["n_rows"].iloc[0]
    all_rows = summary[(summary.readout == "logit_lens")
                       & (summary.condition == "object_high")]["n_rows"].iloc[0]
    assert ps_rows < all_rows, "Patchscopes subset should only count scored rows"
    assert {"object_high", "pooled_mean", "outside", "random", "shuffled"} <= set(
        summary.condition)
    head = E.headline(summary)
    assert set(head.readout) == set(names)
    E.plot(summary, out_dir, run_vtr.MAIN_READOUTS)
    for f in ("accuracy_by_layer.png", "accuracy_by_overlap.png", "pooling.png"):
      assert os.path.exists(os.path.join(out_dir, f)), f
    print("readouts + evaluation + plots OK")

    # run_vtr's eval stage end to end, including the fair-subset outputs.
    import json
    with open(os.path.join(out_dir, "samples.json"), "w", encoding="utf-8") as f:
      json.dump({"class_names": class_names, "geometry": "llava-1.5", "samples": samples}, f)
    torch.save({"states": states, "meta": meta}, os.path.join(out_dir, "states.pt"))
    run_vtr.main(["--out_dir", out_dir, "--stage", "eval", "--n_boot", "50"])
    for f in ("summary.csv", "headline.csv", "fair_classes.json"):
      assert os.path.exists(os.path.join(out_dir, f)), f
    with open(os.path.join(out_dir, "fair_classes.json"), encoding="utf-8") as f:
      fair = json.load(f)
    assert len(fair["kept"]) + len(fair["dropped"]) == len(class_names)
    print(f"eval stage OK (fair subset keeps {fair['kept']})")

    # Task 5 sweep: cells exist, the diagonal equals ordinary same-layer Patchscopes.
    import run_ps_sweep as S
    sweep_dir = os.path.join(out_dir, "sweep")
    S.main(["--vtr_dir", out_dir, "--out_dir", sweep_dir, "--layers", "1,2",
            "--n_images", "4", "--rows_per_batch", "16", "--n_boot", "20",
            "--device", "cpu"], mt=mt)
    with open(os.path.join(sweep_dir, "config.json"), encoding="utf-8") as f:
      rows = np.array(json.load(f)["rows"])
    ps = Patchscopes(mt, class_names, class_forms(class_names, False), "identity", "cpu",
                     rows_per_batch=16)
    _, expected = ps.scores(states[rows, 2].float(), 2)
    got = np.load(os.path.join(sweep_dir, "cells", "s2_t2.npz"))["pmi"].astype(np.float32)
    assert np.allclose(got, expected, atol=2e-2), np.abs(got - expected).max()
    off = np.load(os.path.join(sweep_dir, "cells", "s2_t1.npz"))["pmi"].astype(np.float32)
    assert not np.allclose(off, got, atol=1e-3), "target layer had no effect"
    for f in ("sweep_summary.csv", "sweep_best_target.csv", "sweep_heatmap.png",
              "sweep_summary_fair.csv", "sweep_heatmap_fair.png"):
      assert os.path.exists(os.path.join(sweep_dir, f)), f
    print("source x target sweep OK (diagonal == same-layer Patchscopes)")

  print("\nALL TESTS PASSED")


if __name__ == "__main__":
  main()
