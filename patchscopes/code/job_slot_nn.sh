#!/bin/bash -l
# Image-slot injection read by LatentLens (run_vtr.py slot_nn), with the full Visual Genome bank
# from job_ll_bank.sh. Submit it held behind the bank:  qsub -hold_jid ll_bank job_slot_nn.sh
#   1. plain LatentLens on that bank for every benchmark row  -> scores/latentlens_vg*.npz
#   2. slot_nn on the first 200 images, even layers           -> scores/slot_nn_latentlens_vg_*
#      (inject at the dev-chosen image-slot target t, read at t + 0/2/4/8/16)
#   3. eval stage (adds latentlens_vg to summary_fair.csv), then the same-rows comparison
#      -> slot_nn/compare_latentlens_vg.csv
# Layer-level checkpoints: resubmit to resume.
#$ -P mlresearch
#$ -N slot_nn
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_memory=48G
#$ -pe omp 8
#$ -l h_rt=08:00:00
#$ -j y
#$ -o logs/slot_nn.log

set -e
module load miniconda
conda activate patchscopes
export HF_HOME=/projectnb/mlresearch/vishnuav/hf_cache
cd /projectnb/mlresearch/vishnuav/interpretability/patchscopes/code

BANK=./results/ll_llava15/bank
OUT=./results/vtr_llava15
n=$(find $BANK -path "*layer_*" -name embeddings_cache.pt | wc -l)
if [ "$n" -lt 8 ]; then
  echo "bank incomplete: $n of 8 layer_*/embeddings_cache.pt under $BANK (did ll_bank finish?)"
  exit 1
fi

python run_vtr.py --stage readouts --out_dir $OUT --readouts latentlens \
    --latentlens_bank $BANK --latentlens_name latentlens_vg
python run_vtr.py --stage readouts --out_dir $OUT --readouts slot_nn \
    --layers 0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30 --ps_max_images 200 \
    --latentlens_bank $BANK --latentlens_name latentlens_vg
python run_vtr.py --stage eval --out_dir $OUT
python compare_slot_nn.py --out_dir $OUT --bank_name latentlens_vg
