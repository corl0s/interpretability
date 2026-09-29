# Runbook: LatentLens readout on the object-identification states

Script: `latentlens_object_identification.py` · Test: `test_latentlens_object_identification.py`

Adds LatentLens (Krojer et al., ICML 2026, arXiv 2602.00462) as a fourth readout next to
Patchscopes, logit lens and the linear probe, **on the same cached states**
(`results/object_identification/patch_states.pt`) and with the same ground truth. No images are
re-run; the only GPU work is building the text bank.

## What it does

1. **Builds a contextual bank from LLaVA-1.5's own LLM.** It runs LLaVA's language model over
   LatentLens's corpus (`concepts.txt`, 117k sentences, downloaded automatically) and stores
   the hidden state of each token in context at 8 layers (HF index `1,2,4,8,16,24,30,31`).
   There is no pre-built bank for LLaVA-1.5, and base Vicuna would be the wrong space, because
   LLaVA fine-tunes the LLM.
2. **Queries every cached patch state** (object, outside, and one random control per image)
   at every layer. It returns the top-5 nearest bank entries, merged across bank layers as
   LatentLens does.
3. **Scores** each neighbour three ways:
   - `token`: the logit-lens rule (first sub-token of the class name).
   - `word`: the whole word in the neighbour's source sentence ("ref" → "refrigerator").
   - `word_lenient`: `word` plus the curated synonyms.

   Headline numbers use the same clean classes as `analyze_object_identification.py`.

Bank construction and search are re-implemented in the script instead of `pip install
latentlens`, because PyPI 0.1.0 and GitHub `main` differ and the package needs Python ≥ 3.10.
The test checks the search is **identical** to the official `ContextualIndex.search` (passed
2026-09-29 against the 0.1.0 wheel).

## Layer convention (important when comparing with the paper)

In this project, layer `l` is the output of block `l`, the same convention as the Patchscopes and logit-lens
curves. LatentLens and HF `hidden_states` call that layer `l+1`. The CSV stores both (`layer`,
`hf_layer`). The bank never uses HF index 32, which is post final-norm.

## Running it on SCC

```bash
conda activate patchscopes
cd /projectnb/cs505am/students/vishnuav/interpretability/patchscopes/code

# 0. CPU test (tiny random model; ~1 min)
python test_latentlens_object_identification.py

# 1. corpus coverage of the COCO classes (CPU, seconds)
python latentlens_object_identification.py --stage coverage

# 2. smoke run: 10k-sentence bank (~1 h on one GPU)
python latentlens_object_identification.py --corpus_limit 10000 \
    --index_dir ./results/latentlens_index_smoke --out_dir ./results/latentlens_smoke

# 3. full run (bank ~13 h per LatentLens README; request ~64 GB RAM for the build)
python latentlens_object_identification.py
```

Run `analyze_object_identification.py` first if `class_control_fp.csv` and
`bootstrap_by_layer.csv` are not in `results/object_identification/`. They provide the
clean-class list and the Patchscopes and logit-lens curves for the combined plot.

Disk: the bank is stored in fp16, about 2 GB per layer on the full corpus, so ~16 GB for 8
layers. Check the `/projectnb` quota first.

## Outputs (`results/latentlens/`)

| File | Contents |
|---|---|
| `corpus_class_coverage.csv` | corpus sentences mentioning each class (exact / with synonyms) |
| `latentlens.csv` | per query × layer: nearest token, its word and sentence, bank layer, similarity, top-1/top-5 correctness |
| `latentlens_summary.csv` | per layer, image-clustered 95% CI, every condition |
| `latentlens_headline.json` | layers 2, 29 and best layer |
| `readout_comparison_latentlens.png` | LatentLens + Patchscopes + logit lens + probe on one plot |

## Check these before trusting the numbers

1. **Corpus coverage.** On the full `concepts.txt`, some COCO classes are rare or absent by
   exact name: baseball glove 0, sports ball 0, stop sign 1, frisbee 1, skis 1, wine glass 2,
   surfboard 2, snowboard 2, **dining table 3** (a common class in our set). The median is 31.
   LatentLens can only return words that are in the corpus, so exact `word` scoring
   penalises these classes. Report `word_lenient` next to it, and a per-class breakdown if
   the gap is large. LatentLens's paper used ~3M Visual Genome phrases, not `concepts.txt`.
   A VG-phrase corpus is a possible follow-up if coverage turns out to matter.
2. **Controls.** The `outside` and `random` curves must stay near 0, as they do for Patchscopes.
3. **Spot-check `latentlens.csv`.** Read `nn_word` and `nn_caption` for a few object rows at
   layer 2 and layer 29.
