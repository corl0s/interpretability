#!/bin/bash -l
# Fit label-free rewind maps on 2,000 unused COCO images (GPU, ~30-60 min)
#$ -P mlresearch
#$ -N fit_rewind
#$ -l gpus=1
#$ -l gpu_memory=40G
#$ -l h_rt=02:00:00
#$ -j y
#$ -o logs/fit_rewind.log

module load miniconda
conda activate patchscopes
export HF_HOME=/projectnb/mlresearch/vishnuav/hf_cache
cd /projectnb/mlresearch/vishnuav/interpretability/patchscopes/code

python test_rewind.py && python fit_rewind_maps.py --coco_dir /projectnb/mlresearch/vishnuav/coco \
    --vtr_dir ./results/vtr_llava15
