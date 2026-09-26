#!/bin/bash
#SBATCH --job-name=gen_smoke
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --output=/raid/home/alexzm/scripts/llm-synthetic-data-develop/slurm/logs/%j_gen_smoke.out
#SBATCH --nodelist=dgxa100jal
#SBATCH --partition=dgx_large
set -e

# Mandatory before launching the full run: the whole data flow (generate ->
# mine -> export) over a small slice of the corpus. Proves the answer is not
# empty (the reasoning-mode trap), shows sample triplets, and times the batch so
# the full run can be extrapolated. Writes only under smoke/: nothing is copied
# to the downstream repos. No --gres=gpu: Ollama owns its own GPU, and mining a
# few dozen rows is fine on CPU.

pwd; hostname; date

source /shared/apps/Python/Tensorflow/3.11.6/etc/profile.d/conda.sh
conda activate datagen

cd /raid/home/alexzm/scripts/llm-synthetic-data-develop

CORPUS=corpora/iieg_diverso_2023_2025/md
MODEL=$(awk '/^model_name:/{print $2}' configs/create_embeddings.yaml)

echo "========================================"
echo "  Smoke test: $MODEL over one document per area"
echo "========================================"

curl -sf --max-time 10 http://localhost:11434/api/tags > /dev/null \
  || { echo "ollama no responde en localhost:11434"; exit 1; }
curl -sf --max-time 10 http://localhost:11434/api/tags | grep -q "\"$MODEL\"" \
  || { echo "ollama no tiene el modelo $MODEL"; exit 1; }

# The first ~10 paragraphs of one document per area: economia, sociedad,
# geografia_ambiente, gobierno_seguridad and a municipal booklet.
rm -rf smoke && mkdir -p smoke/md
for doc in expectativas_2024 estructura_demografica_2024 ordenamientos_2024 encig_2023 guadalajara_2025; do
    head -20 "$CORPUS/$doc.md" > "smoke/md/$doc.md"
done

SECONDS=0
# Stacked configs: the second only overrides the paths (this CLI takes no
# --flag overrides; its arguments are positional).
python synthetic.py create_embeddings \
    --config configs/create_embeddings.yaml \
    --config configs/smoke_test.yaml
GEN_S=$SECONDS; echo "tiempo de generacion: ${GEN_S}s"

echo "========================================"
echo "  Mine negatives + export RAG set (tesis env)"
echo "========================================"

conda activate tesis
printf 'input_csv: ./smoke/out/embeddings_qa.csv\noutput_path: ./smoke/mined\n' > smoke/mine.yaml
printf 'input_csv: ./smoke/out/embeddings_qa.csv\noutput_path: ./smoke/ragval\n' > smoke/export.yaml
python synthetic.py mine_negatives --config smoke/mine.yaml
python synthetic.py export_ragval --config smoke/export.yaml

python - "$GEN_S" <<'PY'
import sys
import pandas as pd
gen_s = int(sys.argv[1])
df = pd.read_csv('smoke/mined/embeddings_qa.csv')
rv = pd.read_csv('smoke/ragval/ragval_dataset.csv')
print(f'\naceptadas: {len(df)}   columnas: {list(df.columns)}')
assert df['query'].notna().all() and (df['query'].str.strip() != '').all(), \
    'consultas vacias: el modo thinking sigue activo'
mined = df['hard_negative_mined'].notna().sum() if 'hard_negative_mined' in df else 0
print(f'hard_negative_mined: {mined}/{len(df)}')
print(f'ragval: {len(rv)} preguntas, {rv.chunk_id.nunique()} chunks, {rv.documento.nunique()} documentos')
print(f'tiempo: {gen_s}s para 50 anclas -> ~{gen_s/50*4640/3600:.1f} h para ~4,640 anclas')
print(f'por documento: {df.source_file.value_counts().to_dict()}')
for _, r in df.head(10).iterrows():
    print(f"\n  [{r['source_file']}]\n  Q: {r['query']}\n  A: {str(r['answer'])[:160]}..."
          f"\n  N: {str(r['hard_negative'])[:160]}...\n  M: {str(r.get('hard_negative_mined'))[:160]}...")
PY

pwd; hostname; date
