# Runbook: COCO-VTR Milestone 1 (LLaVA-1.5, objects, all readouts)

Script: `run_vtr.py` · Package: `vtr/` · Test: `test_vtr.py`

## What it does

1. **data:** selects COCO **val2017** images, one target object each:
   - non-crowd, and the only instance of its class in the image
   - area ≥ 4000 px²
   - ≥ 4 tokens with ≥ 50% coverage
   - balanced across classes (≤ 20 images per class, up to 1000 images)

   It stores the **coverage of every token** (24×24 grid, exact CLIP crop geometry).
2. **cache:** stores the hidden state at all 32 layers for:
   - 2 tokens per overlap bin (0–25, 25–50, 50–75, 75–100%) and 2 outside tokens
   - **mean- and max-pooled** object states (tokens ≥ 50%)
   - a norm-matched **random** vector
3. **readouts:** every readout scores the same **80 COCO class names** (closed set):

   | Output name | Readout |
   |---|---|
   | `logit_lens`, `logit_lens_syn` | final norm + lm_head; best first-sub-token log-prob of the name (or of an unambiguous synonym) |
   | `latentlens`, `latentlens_syn` | max cosine to bank entries whose word names the class; uses the LLaVA bank in `results/latentlens_index/bank` |
   | `patchscopes_raw`, `patchscopes_pmi` | inject into the identity prompt (`... ?`) and score the continuation `" -> <class>"`. PMI subtracts the no-injection score, because the prompt itself favours some words (it contains "cat") |
   | `probe_linear`, `probe_mlp` | supervised references (availability), 5 folds with images grouped |
   | `tuned_lens` | logit lens after per-layer affine translators trained on **text** by `train_tuned_lens.py` (Belrose et al. 2023; KL to the model's final distribution) |
   | `embedding_lens` | max cosine to the **input** embedding of the class's first sub-token (LatentLens's EmbeddingLens) |
   | `selfie_raw`, `selfie_pmi` | SelfIE-style prompt `USER: _ _ _ _ _\nASSISTANT: Sure, I'll summarize your message:`, state injected at all five `_`, continuation `" <class>"`. Same-layer injection and closed-set scoring, unlike the original SelfIE |
4. **eval:**
   - top-1 / top-5 per layer and condition, with image-clustered 95% CIs
   - conditions: `object_high` (≥ 50%), each overlap bin, pooled mean/max, and the controls `outside`, `random` and `shuffled` (another image's label)
   - outputs: `summary.csv` (everything), `headline.csv` (best layer, layer 2, onset layer = first layer above all controls), and three plots

The eval stage also writes a **fair** version, `headline_fair.csv`, `summary_fair.csv` and `fair_*.png`. It is restricted to classes that every training-free readout (logit lens, LatentLens, Patchscopes PMI) can score and tell apart, ranked among those classes only. `fair_classes.json` lists the dropped classes with the reason for each. Use the fair version for comparisons between tools.

Ranking is pessimistic: ties with the true class count against the readout, and a class a readout cannot score at all is ranked last.

## Run on SCC

```bash
conda activate patchscopes
export HF_HOME=/projectnb/mlresearch/vishnuav/hf_cache
cd /projectnb/mlresearch/vishnuav/interpretability/patchscopes/code

# 0. CPU test (tiny random model, ~2 min)
python test_vtr.py

# 1. smoke run on GPU (20 images, 6 layers, ~15 min)
python run_vtr.py --coco_dir /projectnb/mlresearch/vishnuav/coco --out_dir ./results/vtr_smoke \
    --n_images 20 --layers 0,2,8,16,24,31 --n_boot 200

# 2. full run: submit as a batch job (several hours, mostly Patchscopes and probes)
python run_vtr.py --coco_dir /projectnb/mlresearch/vishnuav/coco
```

Stages are checkpointed in `--out_dir` (`samples.json`, `states.pt`, `scores/*.npz`). Re-running skips anything finished, so a job that times out can simply be resubmitted. To run one stage only, use `--stage readouts`, or `--readouts probe` for one readout.

## Adding the extra baselines to an existing run

Finished readouts are skipped, so the new ones can be added to `results/vtr_llava15` without
recomputing anything:

```bash
# 1. train the tuned lens on text (~0.5-1 h on one GPU; needs ~25 GB GPU memory)
python train_tuned_lens.py --out ./results/tuned_lens/llava15_tuned_lens.pt
#    check the printed held-out KL: tuned lens < logit lens at every layer except 31 (equal)

# 2. tuned lens + embedding lens on all layers (minutes)
python run_vtr.py --out_dir ./results/vtr_llava15 --stage readouts --readouts tuned_lens embedding_lens

# 3. SelfIE prompt on the same layers as Patchscopes (~2.5 h)
python run_vtr.py --out_dir ./results/vtr_llava15 --stage readouts --readouts selfie \
    --layers 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30 --ps_rows_per_batch 512

# 4. re-evaluate everything
python run_vtr.py --out_dir ./results/vtr_llava15 --stage eval
```

## Cost knobs
- **Patchscopes** dominates: 80 continuations per injected state × ~13k rows × 32 layers. Options:
  - `--ps_rows_per_batch 512` on a large GPU
  - `--ps_max_images 300` to subsample images
  - `--layers 0,2,4,...` to subsample layers
- **Probes:** 32 layers × 5 folds × 2 probe types on CPU. Run separately with `--stage readouts --readouts probe` if needed.

## Check before trusting the numbers
1. `data_summary.json`: number of classes and images per class.
2. The `verify` lines printed for the first image (captured states equal HF hidden states).
3. **Controls:** in `headline.csv`, `max_control` should be near the 1/80 chance level (≈ 1–3%). A high control for a readout means the readout favours some class names regardless of input.
4. LatentLens prints the classes no bank entry names. Those are always ranked last; the `_syn` variant should reduce that list.
