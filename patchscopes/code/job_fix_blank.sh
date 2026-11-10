#!/bin/bash -l
# Re-run everything on LLaVA-1.5-7B that scored classes by first sub-tokens before the
# class_token_ids fix (the bare "▁" token was a first sub-token of 56 classes):
#   1. keep the old lens scores and summaries next to the new ones (*_preblank)
#   2. logit / tuned / embedding lens on the cached states, then eval
#   3. the causal test into causal_fixed/ (the old causal/ is kept for comparison)
#   qsub job_fix_blank.sh
#$ -P mlresearch
#$ -N fix_blank
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_memory=40G
#$ -pe omp 4
#$ -l h_rt=16:00:00
#$ -j y
#$ -o logs/fix_blank.log

set -e
module load miniconda
conda activate patchscopes
export HF_HOME=/projectnb/mlresearch/vishnuav/hf_cache
cd /projectnb/mlresearch/vishnuav/interpretability/patchscopes/code

VTR=./results/vtr_llava15
python test_revisit.py
mkdir -p $VTR/scores_preblank
keep() {  # copy $1 to $2 once (never overwrite the first backup)
  if [ -f "$1" ] && [ ! -f "$2" ]; then cp "$1" "$2"; fi
}
for n in logit_lens tuned_lens embedding_lens; do
  keep $VTR/scores/$n.npz $VTR/scores_preblank/$n.npz
done
for f in headline summary_fair; do
  keep $VTR/$f.csv $VTR/${f}_preblank.csv
done
keep $VTR/fair_classes.json $VTR/fair_classes_preblank.json

python run_vtr.py --stage readouts --out_dir $VTR --overwrite \
    --readouts logit_lens tuned_lens embedding_lens
python run_vtr.py --stage eval --out_dir $VTR
python causal_restore.py --coco_dir /projectnb/mlresearch/vishnuav/coco \
    --vtr_dir $VTR --out_dir $VTR/causal_fixed --n_images 200
