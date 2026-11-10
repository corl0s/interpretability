#!/bin/bash -l
# Revisiting published findings on LLaVA-1.5-7B (one experiment per submission; all resumable):
#   qsub job_revisit.sh maps          A1/A2/A4/A5: every token of 50 images read at layers
#                                     0, 8, 16, 24 (run_token_maps.py)       ~4 h
#   qsub job_revisit.sh pruning       C3: object tokens hidden after layer K vs random / half
#                                     (run_object_pruning.py, 200 images)    ~2 h
#   qsub job_revisit.sh halluc_smoke  D1 on 5 images (run this before halluc)
#   qsub job_revisit.sh halluc        D1: hallucination detection, 500 captions
#                                     (run_hallucination.py)                 ~8 h
#$ -P mlresearch
#$ -N revisit
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_memory=40G
#$ -pe omp 4
#$ -l h_rt=12:00:00
#$ -j y
#$ -o logs/revisit_$JOB_ID.log

set -e
module load miniconda
conda activate patchscopes
export HF_HOME=/projectnb/mlresearch/vishnuav/hf_cache
cd /projectnb/mlresearch/vishnuav/interpretability/patchscopes/code

COCO=/projectnb/mlresearch/vishnuav/coco
VTR=./results/vtr_llava15
python test_revisit.py
case "$1" in
  maps)
    python run_token_maps.py --coco_dir $COCO --vtr_dir $VTR --n_images 50 --layers 0,8,16,24 ;;
  pruning)
    python run_object_pruning.py --coco_dir $COCO --vtr_dir $VTR --n_images 200 ;;
  halluc_smoke)
    python run_hallucination.py --coco_dir $COCO --vtr_dir $VTR --n_images 5 \
        --out_dir $VTR/hallucination_smoke --n_boot 50 ;;
  halluc)
    python run_hallucination.py --coco_dir $COCO --vtr_dir $VTR --n_images 500 ;;
  *)
    echo "usage: qsub job_revisit.sh maps|pruning|halluc_smoke|halluc"; exit 1 ;;
esac
