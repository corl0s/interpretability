#!/bin/bash -l
# Why LatentLens scores lower with the Visual Genome bank than with the concepts bank on COCO-VTR
# (diag_latentlens_bank.py). Cached states only, no LLaVA passes: qsub job_diag_bank.sh
#$ -P mlresearch
#$ -N diag_bank
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_memory=48G
#$ -pe omp 8
#$ -l h_rt=02:00:00
#$ -j y
#$ -o logs/diag_bank.log

set -e
module load miniconda
conda activate patchscopes
export HF_HOME=/projectnb/mlresearch/vishnuav/hf_cache
cd /projectnb/mlresearch/vishnuav/interpretability/patchscopes/code

python diag_latentlens_bank.py --out_dir ./results/vtr_llava15 \
    --ref ./results/latentlens_index/bank:latentlens \
    --new ./results/ll_llava15/bank:latentlens_vg
