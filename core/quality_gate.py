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

# Page headers docling glues onto the passage that follows them: "Página 20 de 40"
# anywhere, or a bare "50 de 91" opening the passage. The bare form only counts
# before a capitalized word, so "5 de 10 personas..." is left alone.
PAGE_HEADER = re.compile(r'P[aá]gina\s+\d+\s+de\s+\d+\s*|^\s*\d{1,3}\s+de\s+\d{1,3}\s+(?=[A-ZÁÉÍÓÚÑ¿])')

# Anchors that are document scaffolding, not content: a table of contents (dot
# leaders), and questionnaire annexes (Word checkbox glyphs and broken
# cross-references).
TOC_LEADER = re.compile(r'\.{4,}|…{2,}')
QUESTIONNAIRE = re.compile('[]|No se encuentra el origen de la referencia')

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
# ponytail: directory heuristic. Measured on the diverse IIEG corpus: under these
# fractions only lists of names (authors, city council) remain; tables and ranked
# lists of municipalities carry figures and pass.
MIN_PROSE_FRACTION = 0.15   # lowercase words among all tokens
MIN_FIGURE_FRACTION = 0.05  # numeric tokens among all tokens


def clean_passage(text: str) -> str:
    """Strip page headers glued onto a passage and normalize Word's private-use
    bullet glyph; the content itself is kept."""
    return PAGE_HEADER.sub('', str(text)).replace('\uf0b7', '•').strip()


def _is_noise(passage: str) -> bool:
    """
    True when the anchor is document scaffolding: a table of contents, a
    questionnaire annex, or a directory of names. A query about such a
    passage cannot be answered from domain content.
    """
    if len(TOC_LEADER.findall(passage)) >= 2 or QUESTIONNAIRE.search(passage):
        return True
    tokens = passage.split()
    if not tokens:
        return True
    prose = sum(bool(re.fullmatch(r'[a-záéíóúñü]{3,}[,.;:]?', t)) for t in tokens) / len(tokens)
    figures = sum(bool(re.fullmatch(r'[\d.,%$()-]+', t)) for t in tokens) / len(tokens)
    return prose < MIN_PROSE_FRACTION and figures < MIN_FIGURE_FRACTION


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
    Beyond the protocol, anchors that are document scaffolding (tables of
    contents, questionnaires, name directories) are rejected as noise.
    Exact duplicate queries are handled by `apply`, which needs the whole frame.
    """
    if not isinstance(query, str) or not query.strip():
        return 'empty'
    if _is_noise(str(answer)):
        return 'noise_anchor'
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

    # Scaffolding anchors found in the diverse IIEG corpus.
    toc = ('Análisis demográfico del área circundante ........................ 3 Del área de '
           'influencia de 500m ........................ 6 Unidades económicas ........ 19')
    form = '13. ¿Ha aumentado los precios en los últimos 3 meses?  Sí  No  No sé'
    names = ('Regidores (as) Ana Isabel Robles Jiménez María Andrea Medrano Ortega Humberto '
             'Gabriel Trujillo Jiménez Leticia Fabiola Cuan Ramírez')
    ranked = ('Zona Metropolitana de Guadalajara 68.4% Puerto Vallarta 6.6% Zapotlán el Grande '
              '3.2% Autlán de Navarro 3.0% Lagos de Moreno 2.2%')
    for noise in (toc, form, names):
        assert reject_reason('Ana Isabel Robles Jiménez Regidores 19 Sí', noise) == 'noise_anchor', noise
    assert reject_reason('participación de Puerto Vallarta 6.6%', ranked) is None

    assert clean_passage('Página 20 de 40 Si bien DiDi Food...') == 'Si bien DiDi Food...'
    assert clean_passage('50 de 91 Con respecto a las ventas') == 'Con respecto a las ventas'
    assert clean_passage('el índice optimista. Página 9 de 98 Con respecto') == \
        'el índice optimista. Con respecto'
    assert clean_passage('5 de 10 personas encuestadas') == '5 de 10 personas encuestadas'
    assert clean_passage('\uf0b7 Jalisco es la tercera entidad') == '• Jalisco es la tercera entidad'

    df = pd.DataFrame({
        'query': ['población de Acatic en 2020', 'Población de Acatic en 2020', '{question}'],
        'answer': [passage] * 3,
    })
    kept, rejected = apply(df)
    assert len(kept) == 1, kept
    assert set(rejected['reject_reason']) == {'duplicate', 'placeholder'}, rejected

    print(report(kept, rejected))
    print('quality_gate self-check OK')
