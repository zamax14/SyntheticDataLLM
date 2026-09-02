# ==============================================================================
# Project Name: Synthetic Data Generator
# Script Name: quality_gate.py
# Description:
#   Automated rejection gate for generated (query, answer) pairs, applied
#   before any dataset is accepted. Implements the five rejection criteria of
#   the thesis protocol (section 10.3), added after the first context-retrieval
#   evaluation had to be discarded: all 17 models scored at chance level
#   because the dataset — not the models — was defective.
# License: MIT
# ==============================================================================

import re
import unicodedata

import pandas as pd

# Template placeholders the generator left unsubstituted, e.g. a literal "{question}".
PLACEHOLDER = re.compile(r'\{[a-z_ ]{2,30}\}', re.IGNORECASE)

# Meta-instructions addressed to the generator, and questions that talk *about*
# the passage instead of asking about its content. Both are non-discriminative:
# they read the same against any passage in the corpus.
META = re.compile(
    r'\b('
    r'fragmento|pasaje|texto anterior|el texto|este texto|el documento|el contexto|'
    r'reformul\w+|parafrase\w+|genera\w*|redacta\w*|escribe|crea\w*|responde en|'
    r'basado en el|seg[uú]n el (?:texto|documento|fragmento|contexto)|'
    r'menciona\w* en|descri\w+ (?:el|la) (?:texto|fragmento)'
    r')\b',
    re.IGNORECASE,
)

STOPWORDS = frozenset("""
a al algo alguna algunas alguno algunos ante antes aquel aquella aquellas aquellos aqui asi aun aunque
cada como con contra cual cuales cuando cuanta cuantas cuanto cuantos de del desde donde dos e el ella
ellas ellos en entre era eran es esa esas ese eso esos esta estan estas este esto estos fue fueron ha
hace hacia han hasta hay la las le les lo los mas me mi mientras muy nada ni no nos o os otra otras otro
otros para pero poco por porque que quien quienes se sea segun ser si sin sobre solo son su sus tal
tambien tan tanto te tiene tienen toda todas todo todos tras un una unas uno unos y ya
""".split())

# ponytail: the two thresholds below are the tuning knobs of the gate. Calibrated
# so the discarded evaluation set is rejected in full; if a legitimate run shows an
# implausible rejection rate, tune these before touching the rules.
MIN_ANCHOR_TOKENS = 2  # content tokens the query must share with its passage
MIN_STRONG_TOKENS = 1  # figures/proper nouns it must share, to be discriminative


def _normalize(text: str) -> str:
    """Lowercase and strip accents, so 'población' and 'poblacion' match."""
    text = unicodedata.normalize('NFD', str(text).lower())
    return ''.join(c for c in text if unicodedata.category(c) != 'Mn')


def _content_tokens(text: str) -> set[str]:
    """Informative tokens: words of 4+ chars or any token containing a digit."""
    tokens = re.findall(r'[a-z0-9]+', _normalize(text))
    return {t for t in tokens if t not in STOPWORDS and (len(t) >= 4 or any(c.isdigit() for c in t))}


def _strong_tokens(passage: str) -> set[str]:
    """
    Tokens that make a query discriminative: figures and proper nouns
    (municipalities, institutions, indicators). Generic vocabulary shared with
    any Spanish text does not distinguish this passage from the rest of the
    corpus — that is precisely how out-of-domain queries slipped into the
    dataset that had to be discarded.
    """
    numbers = {_normalize(t) for t in re.findall(r'\d[\d.,]*', passage)}
    proper = {_normalize(t) for t in re.findall(r'\b[A-ZÁÉÍÓÚÑ][\wÁÉÍÓÚÑáéíóúñ]{3,}', passage)}
    return {t for t in numbers | proper if t and t not in STOPWORDS}


def reject_reason(query: str, answer: str) -> str | None:
    """
    Return the protocol rejection reason for a pair, or None if it passes.

    Criteria (protocol 10.3): unsubstituted template placeholders; meta
    instructions to the generator instead of domain questions; queries with no
    lexical anchoring to their passage — which is also what makes a query
    non-discriminative, i.e. equally applicable to any passage in the corpus,
    and queries anchored only in generic vocabulary, with no figure or proper
    noun from the passage.
    Exact duplicate queries are handled by `apply`, which needs the whole frame.
    """
    if not isinstance(query, str) or not query.strip():
        return 'empty'
    if PLACEHOLDER.search(query):
        return 'placeholder'
    if META.search(query):
        return 'meta_instruction'
    query_tokens = _content_tokens(query)
    if len(query_tokens & _content_tokens(answer)) < MIN_ANCHOR_TOKENS:
        return 'no_anchoring'
    if len(query_tokens & _strong_tokens(answer)) < MIN_STRONG_TOKENS:
        return 'not_discriminative'
    return None


def apply(
    df: pd.DataFrame, query_col: str = 'query', answer_col: str = 'answer'
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a generated dataset into accepted rows and rejected rows.

    Args:
        df: Generated pairs.
        query_col: Column holding the generated query.
        answer_col: Column holding the anchor passage.

    Returns:
        (kept, rejected). `rejected` carries a 'reject_reason' column so the
        rejection rate can be reported by cause, as the protocol requires.
    """
    reasons = [reject_reason(q, a) for q, a in zip(df[query_col], df[answer_col])]
    out = df.assign(reject_reason=reasons)

    # Exact duplicate queries: keep the first occurrence, reject the rest.
    dup = out[query_col].map(_normalize).duplicated(keep='first') & out['reject_reason'].isna()
    out.loc[dup, 'reject_reason'] = 'duplicate'

    kept = out[out['reject_reason'].isna()].drop(columns='reject_reason')
    rejected = out[out['reject_reason'].notna()]
    return kept, rejected


def report(kept: pd.DataFrame, rejected: pd.DataFrame) -> str:
    """One-line-per-cause summary of the rejection rate."""
    total = len(kept) + len(rejected)
    if not total:
        return 'quality gate: no rows'
    lines = [f'quality gate: {len(kept)}/{total} accepted '
             f'({len(rejected) / total:.1%} rejected)']
    lines += [f'  - {reason}: {n}'
              for reason, n in rejected['reject_reason'].value_counts().items()]
    return '\n'.join(lines)


if __name__ == '__main__':
    # Self-check against the three defects actually found in the discarded
    # context-retrieval dataset, plus the duplicate and anchoring criteria.
    passage = ('En 2020 el municipio de Acatic registró una población de 23,241 habitantes, '
               'de los cuales 11,459 son hombres y 11,782 mujeres.')

    assert reject_reason('¿Cuál es el principal objetivo del programa mencionado en el fragmento?',
                         passage) == 'meta_instruction'
    assert reject_reason('Reformula la siguiente pregunta en tres versiones distintas',
                         passage) == 'meta_instruction'
    assert reject_reason('{question}', passage) == 'placeholder'
    assert reject_reason('¿Cuál es la capital de Francia?', passage) == 'no_anchoring'
    assert reject_reason('¿Cómo influye la población de hombres y mujeres en el desarrollo?',
                         passage) == 'not_discriminative'
    assert reject_reason('', passage) == 'empty'
    assert reject_reason('población de Acatic en 2020 por sexo', passage) is None
    assert reject_reason('habitantes de Acatic 23,241', passage) is None

    df = pd.DataFrame({
        'query': ['población de Acatic en 2020', 'Población de Acatic en 2020', '{question}'],
        'answer': [passage] * 3,
    })
    kept, rejected = apply(df)
    assert len(kept) == 1, kept
    assert set(rejected['reject_reason']) == {'duplicate', 'placeholder'}, rejected

    print(report(kept, rejected))
    print('quality_gate self-check OK')
