#!/bin/bash -l
# Same-data control for the oracle: linear probes on the oracle's own training tokens and LLaVA
# crop answers (answer_probe.py), compared with the oracle on its test rows. CPU only.
#   qsub job_answer_probe.sh
#$ -P mlresearch
#$ -N answer_probe
#$ -pe omp 16
#$ -l mem_per_core=8G
#$ -l h_rt=06:00:00
#$ -j y
#$ -o logs/answer_probe.log

set -e
module load miniconda
conda activate patchscopes
cd /projectnb/mlresearch/vishnuav/interpretability/patchscopes/code

python answer_probe.py --vtr_dir ./results/vtr_llava15 \
    --oracle_dir ./results/vtr_llava15/oracle_crop7_add
