#!/bin/bash
#SBATCH --job-name=gen_export
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --output=/raid/home/alexzm/scripts/llm-synthetic-data-develop/slurm/logs/%j_gen_export.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --partition=dgx_large
set -e

pwd; hostname; date

source /shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh
conda activate tesis

cd /raid/home/alexzm/scripts/llm-synthetic-data-develop

echo "========================================"
echo "  Export RAG eval set and distribute datasets"
echo "========================================"

python synthetic.py export_ragval --config configs/export_ragval.yaml

# One generation feeds both downstream repos.
MINED=qa_embeddings_output/mined/embeddings_qa.csv
cp "$MINED" /raid/home/alexzm/scripts/Tesis-Embeddings/datasets/embeddings_qa.csv
cp qa_embeddings_output/ragval/ragval_dataset.csv \
   /raid/home/alexzm/scripts/Tesis-RAG/dataset/ragval_dataset.csv

wc -l "$MINED" qa_embeddings_output/ragval/ragval_dataset.csv

pwd; hostname; date
