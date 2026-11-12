# ==============================================================================
# Project Name: Synthetic Data Generator
# Script Name: multi_tool.py
# Description:
#   Multi-tool requests for the agent evaluation: seeded tool sets of 2-4 tools
#   that form a plausible pipeline, one Spanish request per set written by a
#   generator LLM, and a round-trip check where a third LLM recovers the set
#   from the request alone.
# License: MIT
# ==============================================================================

import random

from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from core.tool_queries import catalog, mentions_tool

CONTEXT = ('Los datos del IIEG están en una base de datos PostgreSQL a la que el agente ya '
           'está conectado.')

REQUEST_PROMPT = """\
Eres usuario del Instituto de Información Estadística y Geográfica de Jalisco (IIEG) y le
escribes a un agente de datos. {context} El agente dispone de estas herramientas:

{catalog}

Escribe UNA petición, en español de México, que para resolverse por completo necesite
exactamente estas herramientas, en este orden, y ninguna otra del catálogo:

{tools}

Reglas:
- Nunca menciones el nombre de una herramienta ni la cites literalmente.
- Cada herramienta debe ser necesaria: la petición pide explícitamente cada resultado
  (un cálculo, una gráfica, un mapa, una tabla, un archivo, la estructura de la base...).
- Que no haga falta ninguna herramienta adicional: si algo se puede resolver con otra
  herramienta del catálogo, no lo pidas.
- Tema concreto del IIEG: {theme}, en Jalisco o sus municipios y regiones.
- Escribe como lo haría una persona real, en una o dos oraciones de 15 a 45 palabras, sin
  listas ni pasos numerados y sin voseo.

Responde solo con JSON: {{"peticion": "..."}}"""

VERIFY_PROMPT = """\
Un agente de datos del IIEG dispone de estas herramientas. {context}

{catalog}

Petición del usuario:
{request}

Indica el conjunto MÍNIMO de herramientas que el agente tiene que llamar para resolver la
petición por completo, en el orden en que las usaría. Incluye solo las estrictamente
necesarias y usa los nombres exactos del catálogo.

Responde solo con JSON: {{"herramientas": ["nombre", ...]}}"""


def tool_catalog(toolval: pd.DataFrame) -> pd.DataFrame:
    """One row per indexed tool of toolval_dataset.csv: name (parsed as the agent
    harness does), chunk_id, description and whether it is a real agent tool."""
    df = toolval.drop_duplicates('chunk_id')
    return pd.DataFrame({
        'name': df['chunk_content'].str.split(':', n=1).str[0].str.strip().values,
        'chunk_id': df['chunk_id'].values,
        'description': df['chunk_content'].values,
        'real': (df['documento'] != 'distractor').values,
    })


def draw(rng: random.Random, stages: dict[str, list[str]], template: list) -> list[str]:
    """One tool set for a template [[stage, min, max], ...], in pipeline order. The
    explore stage keeps its pool order (list -> connect -> schemas -> tables -> info)."""
    tools = []
    for stage, lo, hi in template:
        k = rng.randint(lo, hi)
        picked = rng.sample(stages[stage], k)
        if stage == 'explore':
            picked.sort(key=stages[stage].index)
        tools += [t for t in picked if t not in tools]
    return tools


def valid(tools: list[str], real: set[str], requires: dict[str, list[str]]) -> bool:
    """2-4 tools, at least one real agent tool, and every prerequisite used earlier."""
    return (2 <= len(tools) <= 4 and bool(real & set(tools))
            and all(r in tools[:i] for i, t in enumerate(tools) for r in requires.get(t, [])))


def sample_sets(stages: dict[str, list[str]], templates: list, requires: dict[str, list[str]],
                real: set[str], n: int, sizes: list[int], seed: int,
                max_tries: int = 5000) -> list[list[str]]:
    """n distinct tool sets. Each set is drawn to include the real tool used least so
    far (so every real tool gets covered) with the next size of the `sizes` cycle;
    when no template yields that, first the size and then the target are relaxed."""
    rng = random.Random(seed)
    counts, seen, out = Counter({t: 0 for t in real}), set(), []
    for i in range(n):
        target = min(sorted(real), key=lambda t: (counts[t], rng.random()))
        size = sizes[i % len(sizes)]
        tiers = [lambda s: target in s and len(s) == size, lambda s: target in s, lambda s: True]
        found = next((s for ok in tiers for _ in range(max_tries)
                      for s in [draw(rng, stages, rng.choice(templates))]
                      if ok(s) and valid(s, real, requires) and tuple(s) not in seen), None)
        if found is None:
            raise ValueError(f'only {len(out)} distinct valid sets could be drawn')
        seen.add(tuple(found))
        counts.update(t for t in found if t in real)
        out.append(found)
    return out


def generate_requests(llm, tools: pd.DataFrame, sets: list[list[str]], themes: list[str],
                      seed: int, workers: int) -> pd.DataFrame:
    """One request per set; the theme cycles over `themes` in a seeded order."""
    listing = catalog(tools.sort_values('name'))
    described = dict(zip(tools['name'], tools['description']))
    order = random.Random(seed).sample(range(len(sets)), len(sets))

    def one(i: int) -> dict:
        tool_set = sets[i]
        theme = themes[order[i] % len(themes)]
        prompt = REQUEST_PROMPT.format(
            context=CONTEXT, catalog=listing, theme=theme,
            tools='\n\n'.join(f'{k + 1}. {described[t]}' for k, t in enumerate(tool_set)))
        text = llm(prompt).get('peticion', '')
        return {'set_id': i, 'herramientas': ';'.join(tool_set), 'tema': theme,
                'pregunta': text.strip() if isinstance(text, str) else ''}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return pd.DataFrame(list(pool.map(one, range(len(sets)))))


def names_a_tool(request: str, tool_set: list[str]) -> bool:
    return any(mentions_tool(request, t) for t in tool_set)


def near_copies(queries: list[str], reference: list[str], threshold: float) -> tuple[list, list]:
    """TF-IDF cosine (same vectorizer as build_tool_training) of each query to its
    closest reference request, and whether it is a near-duplicate of an earlier query."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(reference + queries)
    q = vec.transform(queries)
    to_ref = cosine_similarity(q, vec.transform(reference)).max(axis=1) if reference else [0.0] * len(queries)
    within = cosine_similarity(q)
    dup = [bool((within[i, :i] >= threshold).any()) for i in range(len(queries))]
    return [float(x) for x in to_ref], dup


def verify(llm, tools: pd.DataFrame, requests: list[str], seed: int, workers: int) -> list[list[str]]:
    """Round trip: the verifier names the tools each request needs. The catalog is
    shuffled per request (seeded) so list position gives no cue."""
    def one(i: int) -> list[str]:
        shuffled = tools.sample(frac=1, random_state=seed + i)
        out = llm(VERIFY_PROMPT.format(context=CONTEXT, catalog=catalog(shuffled),
                                       request=requests[i])).get('herramientas', [])
        return [str(t).strip() for t in out] if isinstance(out, list) else []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, range(len(requests))))


def jaccard(gold: list[str], predicted: list[str]) -> float:
    a, b = set(gold), set(predicted)
    return len(a & b) / len(a | b) if a | b else 0.0


def pick_threshold(scores: pd.Series, thresholds: list[float], min_kept: int) -> float:
    """Strictest threshold that keeps at least min_kept requests (else the loosest)."""
    for t in sorted(thresholds, reverse=True):
        if (scores >= t).sum() >= min_kept:
            return t
    return min(thresholds)


if __name__ == '__main__':
    stages = {'explore': ['l', 'c', 's'], 'source': ['q'], 'stats': ['d', 'x'],
              'present': ['bar', 'tab'], 'deliver': ['pdf']}
    real = {'l', 'c', 's', 'q', 'd', 'bar', 'tab'}
    requires = {'tab': ['d']}
    templates = [[['source', 1, 1], ['stats', 1, 1], ['present', 0, 1], ['deliver', 0, 1]],
                 [['explore', 1, 3]]]
    rng = random.Random(0)
    for _ in range(200):
        s = draw(rng, stages, templates[1])
        assert s == sorted(s, key=stages['explore'].index)
    assert valid(['q', 'd', 'tab'], real, requires)
    assert not valid(['q', 'tab'], real, requires)            # prerequisite missing
    assert not valid(['tab', 'd'], real, requires)            # prerequisite after
    assert not valid(['x', 'pdf'], real, requires)            # no real tool
    assert not valid(['q'], real, requires) and not valid(['l', 'c', 's', 'q', 'd'], real, {})
    sets = sample_sets(stages, templates, requires, real, 12, [2, 3, 4], seed=1)
    assert sets == sample_sets(stages, templates, requires, real, 12, [2, 3, 4], seed=1)
    assert len({tuple(s) for s in sets}) == 12 and all(valid(s, real, requires) for s in sets)
    assert real <= {t for s in sets for t in s}
    assert names_a_tool('haz un bar chart', ['q', 'bar_chart']) and not names_a_tool('una gráfica', ['bar_chart'])
    to_ref, dup = near_copies(['grafica del empleo en Zapopan', 'grafica del empleo en Zapopan',
                               'mapa de escuelas por municipio'],
                              ['grafica del empleo en Zapopan por mes'], 0.8)
    assert to_ref[0] > 0.5 and to_ref[2] < 0.3 and dup == [False, True, False]
    assert jaccard(['a', 'b'], ['b', 'a']) == 1.0 and jaccard(['a', 'b'], ['a', 'c']) == 1 / 3
    assert jaccard(['a'], []) == 0.0
    scores = pd.Series([1.0] * 5 + [0.75] * 5)
    assert pick_threshold(scores, [1.0, 0.75], 5) == 1.0
    assert pick_threshold(scores, [1.0, 0.75], 8) == 0.75
    assert pick_threshold(scores, [1.0, 0.75], 50) == 0.75
    tv = pd.DataFrame({'chunk_id': ['a1', 'a1', 'b2'], 'documento': ['tablas', 'tablas', 'distractor'],
                       'chunk_content': ['t1: Hace.', 't1: Hace.', 'd1: Otra.']})
    cat = tool_catalog(tv)
    assert cat['name'].tolist() == ['t1', 'd1'] and cat['real'].tolist() == [True, False]
    print('multi_tool self-check OK')
