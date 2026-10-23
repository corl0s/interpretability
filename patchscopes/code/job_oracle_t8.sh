#!/bin/bash -l
# Oracle variant A: crop7 labels (reused from oracle_crop7), injection at target layer 8 instead
# of 4 -> results/vtr_llava15/oracle_crop7_t8. Checkpointed: resubmit to resume.
#$ -P mlresearch
#$ -N oracle_t8
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_memory=40G
#$ -l h_rt=02:30:00
#$ -j y
#$ -o logs/oracle_t8.log

module load miniconda
conda activate patchscopes
export HF_HOME=/projectnb/mlresearch/vishnuav/hf_cache
cd /projectnb/mlresearch/vishnuav/interpretability/patchscopes/code

python train_oracle.py --coco_dir /projectnb/mlresearch/vishnuav/coco \
    --vtr_dir ./results/vtr_llava15 --label_view crop7 --target_layer 8 \
    --data_dir ./results/vtr_llava15/oracle_crop7/data \
    --out_dir ./results/vtr_llava15/oracle_crop7_t8
