#!/bin/bash -l
# Causal check (Task 7) pilot: mask each image's object, restore its clean visual-token states at
# one layer at a time, measure answer recovery, and compare with the saved readout scores
# (causal_restore.py). Resumable: resubmit to continue.
#   qsub job_causal_restore.sh
#$ -P mlresearch
#$ -N causal_restore
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_memory=40G
#$ -l h_rt=03:00:00
#$ -j y
#$ -o logs/causal_restore.log

set -e
module load miniconda
conda activate patchscopes
export HF_HOME=/projectnb/mlresearch/vishnuav/hf_cache
cd /projectnb/mlresearch/vishnuav/interpretability/patchscopes/code

python test_causal_restore.py
python causal_restore.py --coco_dir /projectnb/mlresearch/vishnuav/coco \
    --vtr_dir ./results/vtr_llava15 --n_images 50
