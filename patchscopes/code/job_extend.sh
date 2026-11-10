#!/bin/bash -l
# Image-slot injection, the causal test and object-token pruning on more models. Each model's
# benchmark run (job_vtr_qwen.sh / job_vtr_llava13b.sh, incl. eval) must be finished first.
# Stages, in order (each resumable):
#   qsub job_extend.sh qwen proto     choose image-slot target layers on development images
#   qsub job_extend.sh qwen slot      image-slot readout on the whole benchmark, then eval
#   qsub job_extend.sh qwen causal    mask the object, restore its tokens per layer (200 images)
#   qsub job_extend.sh qwen pruning   object tokens hidden after layer K
#   (the same with 13b instead of qwen)
#$ -P mlresearch
#$ -N extend
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_memory=48G
#$ -pe omp 4
#$ -l h_rt=16:00:00
#$ -j y
#$ -o logs/extend_$JOB_ID.log

set -e
module load miniconda
conda activate patchscopes
export HF_HOME=/projectnb/mlresearch/vishnuav/hf_cache
cd /projectnb/mlresearch/vishnuav/interpretability/patchscopes/code

COCO=/projectnb/mlresearch/vishnuav/coco
case "$1" in
  qwen)  # 28 layers; images capped at 512 tokens as in job_vtr_qwen.sh
    MODEL=Qwen/Qwen2.5-VL-7B-Instruct; VTR=./results/vtr_qwen25; PIX="--max_pixels 401408"
    SOURCES=0,4,8,12,16,20,24,27; TARGETS=0,4,8,12,16
    LAYERS=0,2,4,6,8,10,12,14,16,18,20,22,24,26; AFTER=0,2,4,8,12,16,20,24 ;;
  13b)   # 40 layers
    MODEL=llava-hf/llava-1.5-13b-hf; VTR=./results/vtr_llava13b; PIX=""
    SOURCES=0,4,8,12,16,20,24,28,32,36,39; TARGETS=0,4,8,12,16,20
    LAYERS=0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38
    AFTER=0,2,4,8,12,16,20,24,28,32 ;;
  *) echo "usage: qsub job_extend.sh qwen|13b proto|slot|causal|pruning"; exit 1 ;;
esac

case "$2" in
  proto)
    python run_injection_proto.py --vtr_dir $VTR --out_dir $VTR/injection_proto \
        --model_name $MODEL --variants img_all --sources $SOURCES --targets $TARGETS ;;
  slot)
    python run_vtr.py --stage readouts --out_dir $VTR --model_name $MODEL $PIX \
        --readouts image_slot --layers $LAYERS
    python run_vtr.py --stage eval --out_dir $VTR --model_name $MODEL $PIX ;;
  causal)
    python causal_restore.py --coco_dir $COCO --vtr_dir $VTR --model_name $MODEL $PIX \
        --n_images 200 \
        --readouts logit_lens,embedding_lens,patchscopes_pmi,selfie_pmi,image_slot_pmi,probe_linear ;;
  pruning)
    python run_object_pruning.py --coco_dir $COCO --vtr_dir $VTR --model_name $MODEL $PIX \
        --n_images 200 --after $AFTER ;;
  *) echo "usage: qsub job_extend.sh qwen|13b proto|slot|causal|pruning"; exit 1 ;;
esac
