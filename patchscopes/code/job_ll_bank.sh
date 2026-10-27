#!/bin/bash -l
# LatentLens protocol, part 2: their text bank from LLaVA's own language model on the full
# Visual Genome phrase list (2.99M phrases, layers 1 2 4 8 16 24 30 31, float8, <= 20 contexts
# per token), then the LatentLens search for our cached states. Run after job_ll_prep.sh.
# Their builder resumes from progress.json: if the job hits the time limit, resubmit it.
# Disk: ~26 GB of .npy plus ~80 GB of float32 caches under results/ll_llava15/bank.
#$ -P mlresearch
#$ -N ll_bank
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_memory=48G
#$ -pe omp 8
#$ -l h_rt=36:00:00
#$ -j y
#$ -o logs/ll_bank.log

set -e
module load miniconda
conda activate patchscopes
export HF_HOME=/projectnb/mlresearch/vishnuav/hf_cache
CODE=/projectnb/mlresearch/vishnuav/interpretability/patchscopes/code
OUT=$CODE/results/ll_llava15

cd /projectnb/mlresearch/vishnuav/external/latentlens
python reproduce/scripts/extract_embeddings.py --model $OUT/llava15_lm \
    --layers 1 2 4 8 16 24 30 31 --output-dir $OUT/bank --num-captions -1

cd $CODE
python run_ll_protocol.py --stage latentlens --out_dir $OUT
