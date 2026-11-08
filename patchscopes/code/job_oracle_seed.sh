#!/bin/bash -l
# The visual oracle (job_oracle_add.sh's recipe: crop7 labels, additive injection) with another
# seed, which also draws other held-out classes, another training subset and other eval images,
# then its same-data probe control (answer_probe.py). Same training data shards.
#   qsub job_oracle_seed.sh 1     -> results/vtr_llava15/oracle_crop7_add_s1
# Checkpointed: resubmit to resume.
#$ -P mlresearch
#$ -N oracle_seed
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_memory=40G
#$ -l h_rt=03:00:00
#$ -j y
#$ -o logs/oracle_seed.log

set -e
module load miniconda
conda activate patchscopes
export HF_HOME=/projectnb/mlresearch/vishnuav/hf_cache
cd /projectnb/mlresearch/vishnuav/interpretability/patchscopes/code

SEED=${1:-1}
OUT=./results/vtr_llava15/oracle_crop7_add_s$SEED
python train_oracle.py --coco_dir /projectnb/mlresearch/vishnuav/coco \
    --vtr_dir ./results/vtr_llava15 --label_view crop7 --inject add --seed $SEED \
    --data_dir ./results/vtr_llava15/oracle_crop7/data --out_dir $OUT
python answer_probe.py --vtr_dir ./results/vtr_llava15 --oracle_dir $OUT
