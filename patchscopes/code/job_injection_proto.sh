#!/bin/bash -l
# Injection-variant prototype on LLaVA (GPU, ~2-4 h; cells are checkpointed, resubmit to resume)
#$ -P mlresearch
#$ -N injection_proto
#$ -l gpus=1
#$ -l gpu_memory=40G
#$ -l h_rt=10:00:00
#$ -j y
#$ -o logs/vtr_injection_proto.log

module load miniconda
conda activate patchscopes
export HF_HOME=/projectnb/mlresearch/vishnuav/hf_cache
cd /projectnb/mlresearch/vishnuav/interpretability/patchscopes/code

python test_injection.py && python run_injection_proto.py --vtr_dir ./results/vtr_llava15
