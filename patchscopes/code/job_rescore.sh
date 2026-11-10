#!/bin/bash -l
# B3/A3 (CPU, no model): re-score the saved benchmark scores per object region (Takishita et
# al.'s max-pooling) and on tokens outside the object (rescore_region.py), for every finished
# benchmark run.
#   qsub job_rescore.sh
#$ -P mlresearch
#$ -N rescore
#$ -pe omp 4
#$ -l mem_per_core=8G
#$ -l h_rt=01:00:00
#$ -j y
#$ -o logs/rescore.log

set -e
module load miniconda
conda activate patchscopes
cd /projectnb/mlresearch/vishnuav/interpretability/patchscopes/code

for VTR in ./results/vtr_llava15 ./results/vtr_qwen25 ./results/vtr_llava13b; do
  if [ -f $VTR/fair_classes.json ]; then
    echo "=== $VTR"
    python rescore_region.py --vtr_dir $VTR
  fi
done
