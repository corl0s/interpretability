#!/bin/bash -l
# Causal check, round 2 (causal_restore.py) on the same 200 images as job_causal_restore.sh:
#   restores  object, all_image, context (every token the object does not touch), and at layers
#             0-16 only: k tokens the probe decodes / k it does not / k at random
#   masked_read  read the masked image's tokens with a probe and the logit lens
#   paste        paste the object into another image: what do its tokens and the model say?
# Probes (one per layer used) are fitted first on the cached benchmark states, the 200 images
# left out (~20-30 min on CPU), and cached. -> results/vtr_llava15/causal_more. Resumable.
#$ -P mlresearch
#$ -N causal_more
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_memory=40G
#$ -pe omp 8
#$ -l h_rt=06:00:00
#$ -j y
#$ -o logs/causal_more.log

set -e
module load miniconda
conda activate patchscopes
export HF_HOME=/projectnb/mlresearch/vishnuav/hf_cache
cd /projectnb/mlresearch/vishnuav/interpretability/patchscopes/code

python test_causal_restore.py
python causal_restore.py --coco_dir /projectnb/mlresearch/vishnuav/coco \
    --vtr_dir ./results/vtr_llava15 --out_dir ./results/vtr_llava15/causal_more \
    --n_images ${1:-200} --modes object,all_image,context,decoded,undecoded,random_k \
    --experiments masked_read,paste
