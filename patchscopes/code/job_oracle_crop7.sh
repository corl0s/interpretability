#!/bin/bash -l
# Visual activation oracle with crop7 labels (LLaVA's answer about the 7x7 patch block, cropped
# and enlarged): data -> LoRA training -> eval, in results/vtr_llava15/oracle_crop7.
# Every stage is checkpointed: resubmit to resume.
#$ -P mlresearch
#$ -N oracle_crop7
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_memory=40G
#$ -l h_rt=03:00:00
#$ -j y
#$ -o logs/oracle_crop7.log

module load miniconda
conda activate patchscopes
export HF_HOME=/projectnb/mlresearch/vishnuav/hf_cache
cd /projectnb/mlresearch/vishnuav/interpretability/patchscopes/code

python train_oracle.py --coco_dir /projectnb/mlresearch/vishnuav/coco \
    --vtr_dir ./results/vtr_llava15 --label_view crop7
