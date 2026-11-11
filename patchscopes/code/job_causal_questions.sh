#!/bin/bash -l
# Causal test under another question (causal_restore.py --question): does the layer up to which
# LLaVA uses the object's tokens depend on what it is asked? Same 200 images as causal_fixed
# (seed 0); object tokens only. -> results/vtr_llava15/causal_q_<question>
#   qsub job_causal_questions.sh photo       (photo | describe | yesno)
#$ -P mlresearch
#$ -N causal_q
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_memory=40G
#$ -l h_rt=04:00:00
#$ -j y
#$ -o logs/causal_q_$JOB_ID.log

set -e
module load miniconda
conda activate patchscopes
export HF_HOME=/projectnb/mlresearch/vishnuav/hf_cache
cd /projectnb/mlresearch/vishnuav/interpretability/patchscopes/code

Q=${1:?question: photo, describe or yesno}
python test_causal_restore.py
python causal_restore.py --coco_dir /projectnb/mlresearch/vishnuav/coco \
    --vtr_dir ./results/vtr_llava15 --n_images 200 --modes object --question $Q \
    --out_dir ./results/vtr_llava15/causal_q_$Q
