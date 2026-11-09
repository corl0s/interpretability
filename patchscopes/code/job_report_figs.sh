#!/bin/bash -l
# Figures, tables and number macros for the COCO-VTR report (CPU, no model):
# make_report_figures.py on the finished LLaVA benchmark and causal test. -> results/report
#   qsub job_report_figs.sh
#   qsub job_report_figs.sh 123,456,789,1011     (show these COCO image ids as examples)
#$ -P mlresearch
#$ -N report_figs
#$ -pe omp 4
#$ -l mem_per_core=8G
#$ -l h_rt=01:00:00
#$ -j y
#$ -o logs/report_figs.log

set -e
module load miniconda
conda activate patchscopes
cd /projectnb/mlresearch/vishnuav/interpretability/patchscopes/code

python make_report_figures.py --coco_dir /projectnb/mlresearch/vishnuav/coco \
    --vtr_dir ./results/vtr_llava15 --out_dir ./results/report ${1:+--image_ids $1}
