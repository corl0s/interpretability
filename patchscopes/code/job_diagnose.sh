#!/bin/bash -l
# Where do readouts get their answer from? diagnose_readouts.py on the saved LLaVA scores (CPU):
# outside tokens, easy / hard tokens, what the wrong answers are, readout agreement, distance
# to the object's edge, object size, overlap bins. -> results/vtr_llava15/diagnose
#   qsub job_diagnose.sh                          (LLaVA)
#   qsub job_diagnose.sh ./results/vtr_qwen25     (any other finished benchmark run)
#$ -P mlresearch
#$ -N diagnose
#$ -pe omp 4
#$ -l mem_per_core=8G
#$ -l h_rt=02:00:00
#$ -j y
#$ -o logs/diagnose.log

set -e
module load miniconda
conda activate patchscopes
cd /projectnb/mlresearch/vishnuav/interpretability/patchscopes/code

python test_diagnose_readouts.py
python diagnose_readouts.py --vtr_dir ${1:-./results/vtr_llava15} \
    --coco_dir /projectnb/mlresearch/vishnuav/coco --max_pixels 401408
