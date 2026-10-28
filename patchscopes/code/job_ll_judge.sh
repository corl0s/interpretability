#!/bin/bash -l
# LatentLens protocol, part 3: the open-weight judge (Qwen3-VL-30B-A3B-Instruct-FP8, served by
# vLLM on this GPU) scores every readout with their evaluate_interpretability.py; the 7 readouts
# run as parallel clients so vLLM batches them. Then the summary table.
# Run after job_ll_prep.sh and job_ll_bank.sh. Needs the conda env `vllm` (see the runbook).
#$ -P mlresearch
#$ -N ll_judge
#$ -l gpus=1
#$ -l gpu_c=8.0
#$ -l gpu_memory=48G
#$ -pe omp 8
#$ -l h_rt=06:00:00
#$ -j y
#$ -o logs/ll_judge.log

set -e
module load miniconda
# vLLM JIT-compiles some kernels at startup and needs nvcc (it is only on PATH via the cuda module).
# The vllm env's torch is built for CUDA 13.0, so use a 13.x toolkit (the default module is 12.8).
module load cuda/13.2
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
nvcc --version | tail -n 1
conda activate vllm
export HF_HOME=/projectnb/mlresearch/vishnuav/hf_cache
cd /projectnb/mlresearch/vishnuav/interpretability/patchscopes/code

OUT=./results/ll_llava15
JUDGE=Qwen/Qwen3-VL-30B-A3B-Instruct-FP8
vllm serve $JUDGE --port 8000 --max-model-len 16384 --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt '{"image": 2}' > logs/ll_vllm.log 2>&1 &
SERVER=$!
trap "kill $SERVER" EXIT
until curl -sf http://localhost:8000/v1/models > /dev/null; do
  kill -0 $SERVER || { echo "vLLM server died, see logs/ll_vllm.log"; exit 1; }
  sleep 15
done
echo "judge server up"

for r in logitlens embeddinglens latentlens patchscopes selfie image_slot oracle; do
  python run_ll_protocol.py --stage judge --out_dir $OUT --readout $r --judge_model $JUDGE \
      > logs/ll_judge_$r.log 2>&1 &
done
wait $(jobs -p | grep -v $SERVER) || true
python run_ll_protocol.py --stage summary --out_dir $OUT
