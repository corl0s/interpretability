#!/bin/bash -l
# SelfIE injected at target layer 12, full LLaVA set (GPU, ~2.5 h)
#$ -P mlresearch
#$ -N selfie_t12
#$ -l gpus=1
#$ -l gpu_memory=40G
#$ -l h_rt=06:00:00
#$ -j y
#$ -o logs/vtr_selfie_t12.log

module load miniconda
conda activate patchscopes
export HF_HOME=/projectnb/mlresearch/vishnuav/hf_cache
cd /projectnb/mlresearch/vishnuav/interpretability/patchscopes/code

python run_vtr.py --out_dir ./results/vtr_llava15 --stage readouts --readouts selfie \
    --ps_target_layer 12 --layers 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30 --ps_rows_per_batch 512
