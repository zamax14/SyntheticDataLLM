#!/bin/bash
#SBATCH --job-name=gen_export
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --output=slurm/logs/%j_gen_export.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --partition=dgx_large
set -e

pwd; hostname; date

source "${CONDA_SH:-/shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh}"
conda activate tesis

echo "========================================"
echo "  Export RAG eval set and distribute datasets"
echo "========================================"

python synthetic.py export_ragval --config configs/export_ragval.yaml

# One generation feeds both downstream repos.
MINED=qa_embeddings_output/iieg_diverso_2023_2025/mined/embeddings_qa.csv
cp "$MINED" ../Tesis-Embeddings/datasets/embeddings_qa.csv
cp qa_embeddings_output/iieg_diverso_2023_2025/ragval/ragval_dataset.csv \
   ../Tesis-RAG/dataset/ragval_dataset.csv

wc -l "$MINED" qa_embeddings_output/iieg_diverso_2023_2025/ragval/ragval_dataset.csv

pwd; hostname; date
