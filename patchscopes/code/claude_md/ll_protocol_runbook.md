# LatentLens protocol runbook (LLaVA-1.5)

This puts our readouts and their three lenses into the LatentLens paper's own evaluation, so our numbers sit next to theirs.

**Paper and code:** Krojer et al., ICML 2026, arXiv 2602.00462; code at github.com/McGill-NLP/latentlens.

**Our code:** `run_ll_protocol.py`, with three jobs: `job_ll_prep.sh`, `job_ll_bank.sh` and `job_ll_judge.sh`. The CPU test is `test_ll_protocol.py`.

## What is matched to their code

| Setting | Value |
|---|---|
| Images | PixMo-Cap validation split (`train_test_split(test_size=2048, seed=96817)`), first 100 images |
| Prompt | "Describe this image in detail.", no system prompt |
| Layers | 0, 1, 2, 4, 8, 16, 24, 30, 31 |
| Logit lens | final norm + output head, top-5 raw tokens |
| Embedding lens | cosine with input embeddings, top-5 raw tokens |
| LatentLens text bank | LLaVA's own LM on the full Visual Genome phrase list; layers 1 2 4 8 16 24 30 31; ≤ 20 contexts per token; float8 |
| LatentLens search | cosine, top-5 across all bank layers jointly; sub-words expanded to whole words from the phrase |
| Judge | their prompt (`IMAGE_PROMPT_WITH_CROP`); full image centre-cropped to 512 px with a red 3×3-patch box, plus the crop |
| Interpretable | the judge lists ≥ 1 concrete, abstract or global word |
| Sampling | per layer, 100 patches, one per image, `random.seed(42)`; the judged token is the centre of the 3×3 box |

**Layer convention:** layer *l* is `hidden_states[l]`, the input to block *l* (0 = projector output). COCO-VTR's layer *k* is the output of block *k*, so their *l* = our *k* + 1.

## Our readouts

Each gives the top-5 raw next tokens after injection, in their logit-lens format.

| Readout | Prompt and target |
|---|---|
| Patchscopes | identity prompt, target 0 (best in our sweep) |
| SelfIE | target 12 (best in our sweep) |
| Image-slot injection | all 576 slots, replace, target 8 (best untrained) |
| Oracle | `oracle_crop7_add`: target 4, additive, norm-matched, LoRA on |

Targets use COCO-VTR's convention (the output of block *t*). These readouts are computed only for the planned patches. `plan.json` replays their judge's sampling, and `summary` checks that the judged patches match it (`off_plan` = 0).

## Deviations (report them)

- **Judge:** open-weight Qwen3-VL-30B-A3B-Instruct-FP8 instead of GPT-5, at temperature 0 with `max_tokens` 2048 (theirs is 1000, which cuts off long reasoning before the JSON).
- **Images:** their split was drawn *after* dropping dead URLs. We draw it from the full list with their seed and skip dead URLs, so we get a different random sample from the same distribution.
- **Oracle training layers:** the oracle was trained on COCO-VTR source layers (outputs of blocks 0, 4, …, 28).
- **Validation:** judge-vs-human agreement is not yet measured. The plan is ~150 hand labels, the same check they ran (κ = 0.68).

## One-time setup on SCC

```bash
# their repo, next to interpretability/ (the default --ll_repo)
cd /projectnb/mlresearch/vishnuav && mkdir -p external && cd external
git clone https://github.com/McGill-NLP/latentlens.git

# extra packages in the existing env
module load miniconda && conda activate patchscopes
pip install datasets requests ml_dtypes

# separate env for the judge server (vLLM >= 0.11 for Qwen3-VL)
conda create -n vllm python=3.11 -y && conda activate vllm
pip install vllm openai pandas pillow
```

Check disk space first (`pquota -u mlresearch`). The text bank needs about 110 GB, and the exported LM about 14 GB.

## Run order

```bash
cd /projectnb/mlresearch/vishnuav/interpretability/patchscopes/code
qsub job_ll_prep.sh                                # ~1-2 h
qsub -hold_jid ll_prep job_ll_bank.sh              # ~13+ h; resubmit to resume if it times out
qsub -hold_jid ll_bank job_ll_judge.sh             # ~2-4 h
tail -n 30 logs/ll_judge.log                       # summary table
```

Results go to `results/ll_llava15/`: `summary.csv`, and judge outputs per readout in `judge/<readout>/evaluation_results.json`.

## Sanity check

The paper gives LLaVA-1.5-7B only as a plot: LatentLens averages about 55–62% interpretable, while the embedding and logit lenses are much lower. Our three lenses should land in that range before we read anything into our own readouts.
