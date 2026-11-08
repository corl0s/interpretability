#!/bin/bash -l
# COCO-VTR on Qwen2.5-VL-7B-Instruct, the second model: same images and objects as the LLaVA run
# (--same_images), the training-free lenses that need no Qwen-specific bank, Patchscopes, SelfIE
# and the linear probe. LatentLens needs a Qwen text bank; image-slot / the oracle need a Qwen
# target-layer choice / training (later). Images capped at 512 merged tokens (max_pixels =
# 512 * 28 * 28), a common Qwen setting. Stages are checkpointed: resubmit to resume.
#   qsub job_vtr_qwen.sh smoke      (20 images, 6 layers, separate out_dir: run this first)
#   qsub job_vtr_qwen.sh
#   qsub job_vtr_qwen.sh full 256   (resolution sweep: at most 256 / 1024 tokens per image,
#   qsub job_vtr_qwen.sh full 1024   -> results/vtr_qwen25_t256 / _t1024)
#$ -P mlresearch
#$ -N vtr_qwen
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_memory=40G
#$ -pe omp 4
#$ -l h_rt=12:00:00
#$ -j y
#$ -o logs/vtr_qwen.log

set -e
module load miniconda
conda activate patchscopes
export HF_HOME=/projectnb/mlresearch/vishnuav/hf_cache
cd /projectnb/mlresearch/vishnuav/interpretability/patchscopes/code

TOKENS=${2:-512}
SUFFIX=$([ "$TOKENS" = 512 ] || echo "_t$TOKENS")
ARGS="--coco_dir /projectnb/mlresearch/vishnuav/coco --model_name Qwen/Qwen2.5-VL-7B-Instruct
      --max_pixels $((TOKENS * 784)) --same_images ./results/vtr_llava15/samples.json
      --readouts logit_lens embedding_lens patchscopes selfie probe"
if [ "$1" = "smoke" ]; then
  python run_vtr.py $ARGS --out_dir ./results/vtr_qwen25_smoke$SUFFIX --n_images 20 \
      --layers 0,2,8,16,24,27 --n_boot 200
else
  python run_vtr.py $ARGS --out_dir ./results/vtr_qwen25$SUFFIX
fi
