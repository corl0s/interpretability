#!/bin/bash -l
# Visual activation oracle pilot: data -> LoRA training -> eval (GPU, bf16 needs compute
# capability >= 8.0). Every stage is checkpointed: resubmit (or chain with -hold_jid) to resume.
#$ -P mlresearch
#$ -N oracle_pilot
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_memory=40G
#$ -l h_rt=02:00:00
#$ -j y
#$ -o logs/oracle_pilot.log

module load miniconda
conda activate patchscopes
export HF_HOME=/projectnb/mlresearch/vishnuav/hf_cache
cd /projectnb/mlresearch/vishnuav/interpretability/patchscopes/code

python train_oracle.py --coco_dir /projectnb/mlresearch/vishnuav/coco \
    --vtr_dir ./results/vtr_llava15
