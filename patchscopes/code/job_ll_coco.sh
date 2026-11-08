#!/bin/bash -l
# Judge vs ground truth: the LatentLens protocol (run_ll_protocol.py) on 150 COCO-VTR images
# instead of PixMo-Cap, so every judged patch has known objects. Uses the LLaVA text bank of the
# PixMo run. -> results/ll_coco. Then, in order:
#   qsub job_ll_coco.sh
#   qsub -hold_jid ll_coco job_ll_judge.sh judge ./results/ll_coco
#   (then on the login node)  python run_ll_protocol.py --stage truth --out_dir ./results/ll_coco \
#        --coco_dir /projectnb/mlresearch/vishnuav/coco --n_images 150
#$ -P mlresearch
#$ -N ll_coco
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_memory=48G
#$ -pe omp 8
#$ -l h_rt=06:00:00
#$ -j y
#$ -o logs/ll_coco.log

set -e
module load miniconda
conda activate patchscopes
export HF_HOME=/projectnb/mlresearch/vishnuav/hf_cache
cd /projectnb/mlresearch/vishnuav/interpretability/patchscopes/code

OUT=./results/ll_coco
N="--n_images 150"
python run_ll_protocol.py --stage fetch_coco --out_dir $OUT $N \
    --coco_dir /projectnb/mlresearch/vishnuav/coco --vtr_dir ./results/vtr_llava15
python run_ll_protocol.py --stage extract --out_dir $OUT $N
python run_ll_protocol.py --stage latentlens --out_dir $OUT $N \
    --bank_dir ./results/ll_llava15/bank
python run_ll_protocol.py --stage plan --out_dir $OUT $N
python run_ll_protocol.py --stage ours --out_dir $OUT $N \
    --oracle_dir ./results/vtr_llava15/oracle_crop7_add
