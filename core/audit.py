# ==============================================================================
# Project Name: Synthetic Data Generator
# Script Name: audit.py
# Authors: Alejandro Zarate
# Description:
#   Data audit of the synthetic question sets: lexical anchoring, copied figures,
#   diversity (near duplicates, question openings, length), document coverage,
#   train/test near duplicates and suspicious mined negatives. Read-only.
# License: MIT
# ==============================================================================

import re
from collections import Counter

import numpy as np
import pandas as pd

from core.paraphrase import content_tokens, coverage

NUMBER = re.compile(r'\d[\d.,]*\d|\d')


def lexical(df: pd.DataFrame) -> dict:
    """Share of question words found in the passage, and share of questions that
    copy a figure appearing in the passage."""
    cov = np.array([coverage(q, a) for q, a in zip(df['query'], df['answer'])])
    fig = np.mean([bool(set(NUMBER.findall(str(q))) & set(NUMBER.findall(str(a))))
                   for q, a in zip(df['query'], df['answer'])])
    return {'cobertura_media': cov.mean(), 'cobertura>=0.8': (cov >= 0.8).mean(),
            'copia_cifra': fig}


def diversity(df: pd.DataFrame, threshold: float = 0.8) -> dict:
    """Near-duplicate questions (TF-IDF cosine), top question openings and length."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    q = df['query'].astype(str).tolist()
    sim = cosine_similarity(TfidfVectorizer(sublinear_tf=True).fit_transform(q))
    np.fill_diagonal(sim, 0)
    openings = Counter(' '.join(x.lower().lstrip('¿').split()[:2]) for x in q)
    top = openings.most_common(5)
    return {'casi_duplicadas': (sim.max(axis=1) >= threshold).mean(),
            'palabras_media': np.mean([len(x.split()) for x in q]),
            'top5_inicios': '; '.join(f'{k} ({v / len(q):.0%})' for k, v in top),
            'share_top5_inicios': sum(v for _, v in top) / len(q)}


def coverage_by_document(df: pd.DataFrame) -> dict:
    """How concentrated the questions are in a few documents."""
    counts = df['source_file'].value_counts()
    return {'documentos': len(counts), 'share_top10_docs': counts.head(10).sum() / len(df)}


def cross_near_duplicates(train: pd.DataFrame, test: pd.DataFrame, threshold: float = 0.8) -> float:
    """Share of test questions with a train question above the TF-IDF cosine threshold."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vec = TfidfVectorizer(sublinear_tf=True).fit(pd.concat([train['query'], test['query']]).astype(str))
    sim = cosine_similarity(vec.transform(test['query'].astype(str)), vec.transform(train['query'].astype(str)))
    return float((sim.max(axis=1) >= threshold).mean())


def suspicious_negatives(df: pd.DataFrame, col: str = 'hard_negative_mined') -> float:
    """Proxy of false negatives: share of rows whose mined negative shares at least as
    many content words with the question as the positive does."""
    rows = df.dropna(subset=[col])
    worse = [len(content_tokens(q) & content_tokens(n)) >= len(content_tokens(q) & content_tokens(a))
             for q, a, n in zip(rows['query'], rows['answer'], rows[col])]
    return float(np.mean(worse)) if worse else float('nan')


if __name__ == '__main__':
    toy = pd.DataFrame({'query': ['¿Cuántos patrones había en 2024?', '¿Cuántos patrones había en 2024?', 'empresas en el seguro social'],
                        'answer': ['En 2024 había 100 patrones', 'En 2024 había 100 patrones', 'En 2024 había 100 patrones'],
                        'hard_negative_mined': ['había patrones en 2024', 'nada que ver', None],
                        'source_file': ['a', 'a', 'b']})
    lx = lexical(toy)
    assert lx['copia_cifra'] == 2 / 3 and lx['cobertura>=0.8'] == 2 / 3
    assert diversity(toy)['casi_duplicadas'] == 2 / 3
    assert suspicious_negatives(toy) == 0.5
    assert coverage_by_document(toy)['documentos'] == 2
    print('audit self-check OK')
