#!/bin/bash -l
# Image-slot injection and the visual activation oracle on the full COCO-VTR benchmark (all
# 1,000 images, every cached row incl. controls), at the layers of the Patchscopes / SelfIE runs
# (0, 2, ..., 30), then run_vtr's eval stage: summary_fair.csv gets image_slot_* and oracle_*
# next to the lenses. The fair class set is unchanged (defined by the training-free tools only).
# Image-slot target per source layer = the one chosen on development images in
# results/vtr_llava15/injection_proto/proto_best.csv. Layer-level checkpoints: resubmit to resume.
#$ -P mlresearch
#$ -N full_ours
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_memory=40G
#$ -pe omp 4
#$ -l h_rt=08:00:00
#$ -j y
#$ -o logs/full_ours.log

set -e
module load miniconda
conda activate patchscopes
export HF_HOME=/projectnb/mlresearch/vishnuav/hf_cache
cd /projectnb/mlresearch/vishnuav/interpretability/patchscopes/code

python run_vtr.py --stage readouts --out_dir ./results/vtr_llava15 \
    --readouts image_slot oracle --layers 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30 \
    --oracle_dir ./results/vtr_llava15/oracle_crop7_add
python run_vtr.py --stage eval --out_dir ./results/vtr_llava15
