#!/bin/bash
#SBATCH --job-name=gen_tools
#SBATCH --cpus-per-task=4
#SBATCH --mem=8gb
#SBATCH --output=/raid/home/alexzm/scripts/llm-synthetic-data-develop/slurm/logs/%j_gen_tools.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --partition=dgx_large
set -e

# Usage: sbatch slurm/create_tool_dataset.sh [config]   (smoke: configs/smoke_tool_dataset.yaml)

# No --gres=gpu on purpose: generation talks over HTTP to the `ollama serve`
# daemon already running on this node, which manages its own GPU outside SLURM.

pwd; hostname; date

source /shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh
conda activate datagen

cd /raid/home/alexzm/scripts/llm-synthetic-data-develop

echo "========================================"
echo "  Generate the tool-retrieval set (E3)"
echo "========================================"

curl -sf --max-time 10 http://localhost:11434/api/tags > /dev/null \
  || { echo "ollama no responde en localhost:11434"; exit 1; }

python synthetic.py create_tool_dataset --config "${1:-configs/create_tool_dataset.yaml}"

pwd; hostname; date
