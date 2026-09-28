#!/bin/bash
#SBATCH --job-name=gen_control
#SBATCH --cpus-per-task=4
#SBATCH --mem=8gb
#SBATCH --output=/raid/home/alexzm/scripts/llm-synthetic-data-develop/slurm/logs/%j_gen_control.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --partition=dgx_large
set -e

# Usage: sbatch slurm/create_control_queries.sh [--config overrides.yaml]
#   smoke: --config configs/smoke_control_queries.yaml

# No --gres=gpu on purpose: generation talks over HTTP to the `ollama serve`
# daemon already running on this node, which manages its own GPU outside SLURM.

pwd; hostname; date

source /shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh
conda activate datagen

cd /raid/home/alexzm/scripts/llm-synthetic-data-develop

echo "========================================"
echo "  Generator-bias control queries"
echo "========================================"

curl -sf --max-time 10 http://localhost:11434/api/tags > /dev/null \
  || { echo "ollama no responde en localhost:11434"; exit 1; }

python synthetic.py create_control_queries --config configs/control_queries.yaml "$@"

pwd; hostname; date
