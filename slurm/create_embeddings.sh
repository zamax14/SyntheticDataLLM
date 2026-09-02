#!/bin/bash
#SBATCH --job-name=gen_qa
#SBATCH --cpus-per-task=16
#SBATCH --mem=32gb
#SBATCH --output=/raid/home/alexzm/scripts/llm-synthetic-data-develop/slurm/logs/%j_gen_qa.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --partition=dgx_large
set -e

# No --gres=gpu on purpose: generation talks over HTTP to the `ollama serve`
# daemon already running on this node, which manages its own GPU outside SLURM.

pwd; hostname; date

source /shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh
conda activate datagen

cd /raid/home/alexzm/scripts/llm-synthetic-data-develop

echo "========================================"
echo "  Generate (query, answer, hard_negative) triplets"
echo "========================================"

curl -sf --max-time 10 http://localhost:11434/api/tags > /dev/null \
  || { echo "ollama no responde en localhost:11434"; exit 1; }

python synthetic.py create_embeddings --config configs/create_embeddings.yaml

pwd; hostname; date
