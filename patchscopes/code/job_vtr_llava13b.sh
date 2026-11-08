#!/bin/bash -l
# COCO-VTR on LLaVA-1.5-13B (scale): same images and objects as the 7B run, the training-free
# lenses that need no model-specific bank, Patchscopes, SelfIE and the linear probe (40 layers).
# Stages are checkpointed: resubmit to resume.
#   qsub job_vtr_llava13b.sh smoke   (20 images, 6 layers, separate out_dir: run this first)
#   qsub job_vtr_llava13b.sh
#$ -P mlresearch
#$ -N vtr_13b
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_memory=48G
#$ -pe omp 4
#$ -l h_rt=16:00:00
#$ -j y
#$ -o logs/vtr_13b.log

set -e
module load miniconda
conda activate patchscopes
export HF_HOME=/projectnb/mlresearch/vishnuav/hf_cache
cd /projectnb/mlresearch/vishnuav/interpretability/patchscopes/code

ARGS="--coco_dir /projectnb/mlresearch/vishnuav/coco --model_name llava-hf/llava-1.5-13b-hf
      --same_images ./results/vtr_llava15/samples.json
      --readouts logit_lens embedding_lens patchscopes selfie probe"
if [ "$1" = "smoke" ]; then
  python run_vtr.py $ARGS --out_dir ./results/vtr_llava13b_smoke --n_images 20 \
      --layers 0,2,10,20,30,39 --n_boot 200
else
  python run_vtr.py $ARGS --out_dir ./results/vtr_llava13b
fi
