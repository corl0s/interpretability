#!/bin/bash -l
# Unified decoder with SelfIE at target 12 (CPU, minutes). Submit with -hold_jid selfie_t12.
#$ -P mlresearch
#$ -N unified_selfie_t12
#$ -pe omp 4
#$ -l h_rt=01:00:00
#$ -j y
#$ -o logs/vtr_unified_selfie_t12.log

module load miniconda
conda activate patchscopes
cd /projectnb/mlresearch/vishnuav/interpretability/patchscopes/code

python run_unified_decoder.py --vtr_dir ./results/vtr_llava15 \
    --tools selfie_pmi selfie_t12_pmi latentlens logit_lens --out_dir ./results/vtr_llava15/unified_selfie_t12
