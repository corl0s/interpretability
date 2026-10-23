#!/bin/bash -l
# Oracle teacher accuracy (measure_teacher.py): LLaVA's answer about one token's region (box /
# crop / mask views) vs COCO. Views are saved as they finish: resubmit to resume.
#$ -P mlresearch
#$ -N teacher
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_memory=40G
#$ -l h_rt=01:00:00
#$ -j y
#$ -o logs/teacher.log

module load miniconda
conda activate patchscopes
export HF_HOME=/projectnb/mlresearch/vishnuav/hf_cache
cd /projectnb/mlresearch/vishnuav/interpretability/patchscopes/code

python measure_teacher.py --coco_dir /projectnb/mlresearch/vishnuav/coco \
    --vtr_dir ./results/vtr_llava15
