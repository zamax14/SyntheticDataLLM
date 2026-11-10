# ==============================================================================
# Project Name: Synthetic Data Generator
# Script Name: agent_testset.py
# Description:
#   Test set of the agent evaluation, drawn only from the test partition:
#   document questions in their original and paraphrased wording (paired), with
#   a reference answer written from the gold passage, plus tool requests sampled
#   evenly per tool.
# License: MIT
# ==============================================================================

import hashlib

from concurrent.futures import ThreadPoolExecutor

import pandas as pd

REFERENCE_PROMPT = """\
Pasaje de un documento del IIEG:
{passage}

Pregunta:
{question}

Responde la pregunta usando SOLO el pasaje, en una o dos oraciones en español. Extrae además:
- dato: la cifra o el hecho que responde la pregunta, tal como aparece en el pasaje;
- periodo: el año, mes o periodo al que se refiere ("" si el pasaje no lo dice);
- unidad: la unidad del dato (personas, pesos, %, etc.; "" si no aplica);
- fuente: la fuente que cita el pasaje ("" si no cita ninguna).

Responde solo con JSON: {{"respuesta": "...", "dato": "...", "periodo": "...", "unidad": "...", "fuente": "..."}}"""

REFERENCE_FIELDS = ['respuesta', 'dato', 'periodo', 'unidad', 'fuente']


def chunk_id(text: str) -> str:
    """Same id as the Tesis-RAG evaluation sets: first 12 hex chars of the SHA-256."""
    return hashlib.sha256(str(text).encode()).hexdigest()[:12]


def allocate(sizes: pd.Series, n: int) -> pd.Series:
    """Proportional allocation of n over groups (largest remainder), capped at each group's size."""
    if n > sizes.sum():
        raise ValueError(f'cannot draw {n} of {sizes.sum()}')
    exact = sizes / sizes.sum() * n
    alloc = exact.astype(int)
    for g in (exact - alloc).sort_values(ascending=False, kind='stable').index:
        if alloc.sum() == n:
            break
        if alloc[g] < sizes[g]:
            alloc[g] += 1
    while alloc.sum() < n:  # ponytail: caps can leave a remainder; give it to groups with room
        g = (sizes - alloc).idxmax()
        alloc[g] += 1
    return alloc


def sample_documents(paraphrases: pd.DataFrame, n: int, seed: int, max_coverage: float) -> pd.DataFrame:
    """n test questions whose paraphrase was kept, stratified by source_file; two rows each."""
    kept = paraphrases[paraphrases['answerable'].astype(bool)
                       & (paraphrases['coverage_paraphrase'] <= max_coverage)]
    alloc = allocate(kept.groupby('source_file').size(), n)
    picked = pd.concat([g.sample(alloc[doc], random_state=seed)
                        for doc, g in kept.groupby('source_file') if alloc[doc]])
    picked = picked.sort_values(['source_file', 'query']).reset_index(drop=True)
    picked['par_id'] = [f'd{i:03d}' for i in range(1, len(picked) + 1)]
    rows = []
    for variant, col in [('original', 'query'), ('parafrasis', 'paraphrase')]:
        rows.append(pd.DataFrame({
            'par_id': picked['par_id'], 'variante': variant, 'pregunta': picked[col],
            'pregunta_original': picked['query'], 'chunk_id': picked['answer'].map(chunk_id),
            'chunk_content': picked['answer'], 'documento': picked['source_file'],
        }))
    return pd.concat(rows).sort_values(['par_id', 'variante']).reset_index(drop=True)


def sample_tools(toolval: pd.DataFrame, per_tool: int, seed: int) -> pd.DataFrame:
    """per_tool test requests per agent tool (rows of the E3 set that have a request)."""
    q = toolval[toolval['pregunta'].fillna('').str.strip() != ''].copy()
    q['herramienta'] = q['chunk_content'].str.split(':', n=1).str[0].str.strip()
    small = q.groupby('herramienta').size().loc[lambda s: s < per_tool]
    if not small.empty:
        raise ValueError(f'tools with fewer than {per_tool} requests: {small.to_dict()}')
    out = pd.concat([g.sample(per_tool, random_state=seed) for _, g in q.groupby('herramienta')])
    out = out.sort_values(['herramienta', 'pregunta']).reset_index(drop=True)
    out['id'] = [f'h{i:03d}' for i in range(1, len(out) + 1)]
    return out[['id', 'pregunta', 'chunk_id', 'herramienta']]


def add_references(llm, docs: pd.DataFrame, workers: int) -> pd.DataFrame:
    """One reference answer per pair, written from the gold passage and the original question."""
    pairs = docs.drop_duplicates('par_id')

    def one(row) -> dict:
        out = llm(REFERENCE_PROMPT.format(passage=row.chunk_content, question=row.pregunta_original))
        return {'par_id': row.par_id, **{f: str(out.get(f, '') or '').strip() for f in REFERENCE_FIELDS}}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        refs = pd.DataFrame(list(pool.map(one, pairs.itertuples())))
    refs = refs.rename(columns={'respuesta': 'respuesta_ref'})
    docs = docs.merge(refs, on='par_id', how='left')
    docs.insert(0, 'id', [f'{p}{"o" if v == "original" else "p"}' for p, v in zip(docs.par_id, docs.variante)])
    return docs


if __name__ == '__main__':
    assert allocate(pd.Series({'a': 6, 'b': 3, 'c': 1}), 5).to_dict() == {'a': 3, 'b': 2, 'c': 0}
    assert allocate(pd.Series({'a': 1, 'b': 9}), 4).sum() == 4

    para = pd.DataFrame({
        'query': [f'q{i}' for i in range(8)], 'paraphrase': [f'p{i}' for i in range(8)],
        'answer': [f'pasaje {i}' for i in range(8)], 'source_file': list('aaaabbbc'),
        'answerable': [True] * 7 + [False], 'coverage_paraphrase': [0.1] * 6 + [0.9, 0.1],
    })
    docs = sample_documents(para, 4, seed=1, max_coverage=0.5)
    assert len(docs) == 8 and docs.groupby('par_id').variante.nunique().eq(2).all()
    assert set(docs.pregunta_original) <= {'q0', 'q1', 'q2', 'q3', 'q4', 'q5'}  # q6, q7 not kept

    class Fake:
        def __call__(self, prompt):
            return {'respuesta': 'Hay 100.', 'dato': '100', 'periodo': '2024', 'unidad': 'personas'}
    docs = add_references(Fake(), docs, 1)
    assert docs.respuesta_ref.eq('Hay 100.').all() and docs.fuente.eq('').all() and docs.id.is_unique

    tv = pd.DataFrame({'pregunta': ['x', 'y', 'z', None], 'chunk_id': ['1', '1', '2', '3'],
                       'chunk_content': ['sum: a', 'sum: a', 'plot: b', 'dist: c']})
    try:
        sample_tools(tv, 2, seed=1)
    except ValueError:
        pass
    else:
        raise AssertionError('a tool with too few requests must fail')
    assert sample_tools(tv, 1, seed=1).herramienta.tolist() == ['plot', 'sum']
    print('agent_testset self-check OK')
