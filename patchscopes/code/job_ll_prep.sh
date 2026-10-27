#!/bin/bash -l
# LatentLens protocol, part 1 (see run_ll_protocol.py): PixMo-Cap images, LLaVA states at their
# layers, logit / embedding lens, judge sampling plan, our readouts on the planned patches, and
# LLaVA's language model saved for the text bank (job_ll_bank.sh).
# -> results/ll_llava15. Needs the LatentLens repo cloned to ../../../external/latentlens.
#$ -P mlresearch
#$ -N ll_prep
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_memory=40G
#$ -pe omp 4
#$ -l h_rt=03:00:00
#$ -j y
#$ -o logs/ll_prep.log

set -e
module load miniconda
conda activate patchscopes
export HF_HOME=/projectnb/mlresearch/vishnuav/hf_cache
cd /projectnb/mlresearch/vishnuav/interpretability/patchscopes/code

OUT=./results/ll_llava15
python run_ll_protocol.py --stage fetch --out_dir $OUT
python run_ll_protocol.py --stage extract --out_dir $OUT
python run_ll_protocol.py --stage plan --out_dir $OUT
python run_ll_protocol.py --stage ours --out_dir $OUT \
    --oracle_dir ./results/vtr_llava15/oracle_crop7_add
python run_ll_protocol.py --stage export_lm --out_dir $OUT
