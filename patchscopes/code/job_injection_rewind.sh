#!/bin/bash -l
# Rewind variants of the injection prototype (GPU; cells are checkpointed, resubmit to resume)
#$ -P mlresearch
#$ -N injection_rewind
#$ -l gpus=1
#$ -l gpu_memory=40G
#$ -l h_rt=02:00:00
#$ -j y
#$ -o logs/injection_rewind.log

module load miniconda
conda activate patchscopes
export HF_HOME=/projectnb/mlresearch/vishnuav/hf_cache
cd /projectnb/mlresearch/vishnuav/interpretability/patchscopes/code

# img_all / selfie_text cells already exist (skipped); they are listed for the comparison table.
python run_injection_proto.py --vtr_dir ./results/vtr_llava15 \
    --variants selfie_text img_all selfie_text_rewind img_all_rewind
