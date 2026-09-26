# ==============================================================================
# Project Name: Synthetic Data Generator
# Script Name: tool_queries.py
# Authors: Alejandro Zarate
# Description:
#   Tool-retrieval evaluation set (E3): Spanish requests whose best tool is one
#   of the agent's MCP tools, plus distractor tools that compete in the index.
# License: MIT
# ==============================================================================

import json
import re

from concurrent.futures import ThreadPoolExecutor

import pandas as pd

QUERY_PROMPT = """\
Eres analista del Instituto de Información Estadística y Geográfica de Jalisco (IIEG).
Trabajas con un agente de datos que dispone de estas herramientas:

{catalog}

Escribe {n} peticiones distintas, en español, que un usuario del IIEG le haría al agente
y para las que la MEJOR herramienta sea exactamente esta:

{target}

Reglas:
- Nunca menciones el nombre de la herramienta ni la cites literalmente.
- La herramienta debe ser claramente la mejor opción frente a las demás, en especial frente
  a estas, que son parecidas: {siblings}.
- Mezcla peticiones explícitas ("haz una gráfica de...") con intenciones implícitas
  ("quiero ver cómo evolucionó...").
- Varía los temas del IIEG: empleo formal, exportaciones, población, pobreza, seguridad,
  precios, turismo, vivienda, municipios y regiones de Jalisco.
- Una sola oración por petición, de 8 a 30 palabras.

Responde solo con JSON: {{"peticiones": ["...", "..."]}}"""

DISTRACTOR_PROMPT = """\
Diseñas herramientas para agentes de IA. Un agente ya tiene estas herramientas:

{catalog}

Inventa {n} herramientas NUEVAS del tema "{theme}". Ninguna puede hacer lo mismo que una
de las herramientas existentes. Descríbelas en español, con el mismo estilo que las
existentes: nombre en snake_case, una oración que diga qué hace y una sección Args.

Responde solo con JSON:
{{"herramientas": [{{"name": "...", "description": "Qué hace.\\n\\nArgs:\\n    arg: ..."}}]}}"""


class ToolLLM:
    """Thin client for an OpenAI-compatible server that returns parsed JSON."""

    def __init__(self, model_name: str, base_url: str | None, api_key: str | None,
                 disable_thinking: bool, temperature: float, max_new_tokens: int,
                 seed: int | None):
        # Imported here so the rest of the repo does not need the package.
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.kwargs = {
            'model': model_name, 'temperature': temperature, 'max_tokens': max_new_tokens,
            'response_format': {'type': 'json_object'},
        }
        if seed is not None:
            self.kwargs['seed'] = seed
        if disable_thinking:
            self.kwargs['extra_body'] = {'reasoning_effort': 'none'}

    def __call__(self, prompt: str) -> dict:
        response = self.client.chat.completions.create(
            messages=[{'role': 'user', 'content': prompt}], **self.kwargs
        )
        return parse_json(response.choices[0].message.content or '')


def parse_json(text: str) -> dict:
    """First JSON object in the text; {} when there is none."""
    match = re.search(r'\{.*\}', text, re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def catalog(tools: pd.DataFrame) -> str:
    """One line per tool: name and the first line of its description."""
    return '\n'.join(
        f'- {name}: {desc.split(":", 1)[1].strip().splitlines()[0]}'
        for name, desc in zip(tools['name'], tools['description'])
    )


def mentions_tool(query: str, name: str) -> bool:
    """True if the query names the tool (as identifier or with spaces)."""
    q = query.casefold()
    return name.casefold() in q or name.replace('_', ' ').casefold() in q


def clean_queries(queries: list, name: str) -> list[str]:
    """Drop non-strings, empties, exact duplicates and queries that name the tool."""
    seen, kept = set(), []
    for q in queries:
        if not isinstance(q, str) or not q.strip():
            continue
        q = q.strip()
        if q.casefold() in seen or mentions_tool(q, name):
            continue
        seen.add(q.casefold())
        kept.append(q)
    return kept


def generate_queries(llm: ToolLLM, tools: pd.DataFrame, n: int, workers: int) -> pd.DataFrame:
    """n requests per tool; one row per kept request."""
    listing = catalog(tools)

    def one(row) -> list[dict]:
        siblings = ', '.join(tools[(tools.family == row.family) & (tools.name != row.name)].name)
        prompt = QUERY_PROMPT.format(catalog=listing, n=n, target=row.description,
                                     siblings=siblings or 'ninguna')
        queries = clean_queries(llm(prompt).get('peticiones', []), row.name)
        return [{'query': q, 'answer': row.description, 'source_file': row.family,
                 'tool': row.name, 'kind': 'positive'} for q in queries]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = [r for batch in pool.map(one, tools.itertuples()) for r in batch]
    return pd.DataFrame(rows)


def generate_distractors(llm: ToolLLM, tools: pd.DataFrame, themes: dict[str, list[str]],
                         per_theme: int, workers: int) -> pd.DataFrame:
    """Distractor tools per theme; names never collide with the agent's or each other."""
    listing = catalog(tools)
    jobs = [(kind, theme) for kind, names in themes.items() for theme in names]

    def one(job) -> list[dict]:
        kind, theme = job
        out = llm(DISTRACTOR_PROMPT.format(catalog=listing, n=per_theme, theme=theme))
        return [{**t, 'kind': f'distractor_{kind}', 'theme': theme}
                for t in out.get('herramientas', [])
                if isinstance(t, dict) and t.get('name') and t.get('description')]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        found = [t for batch in pool.map(one, jobs) for t in batch]

    taken = set(tools['name'])
    rows = []
    for t in found:
        name = str(t['name']).strip()
        if name in taken:
            continue
        taken.add(name)
        rows.append({'query': None, 'answer': f"{name}: {str(t['description']).strip()}",
                     'source_file': 'distractor', 'tool': name, 'kind': t['kind'],
                     'theme': t['theme']})
    return pd.DataFrame(rows)


def index_rows(tools: pd.DataFrame) -> pd.DataFrame:
    """One query-less row per agent tool, so a tool stays indexed even if review
    drops every one of its requests."""
    return pd.DataFrame({'query': None, 'answer': tools['description'],
                         'source_file': tools['family'], 'tool': tools['name'],
                         'kind': 'index'})


if __name__ == '__main__':
    assert parse_json('bla {"peticiones": ["a"]} bla') == {'peticiones': ['a']}
    assert parse_json('sin json') == {} and parse_json('{roto') == {}
    assert mentions_tool('Usa bar_chart para esto', 'bar_chart')
    assert mentions_tool('haz un bar chart', 'bar_chart')
    assert not mentions_tool('haz una gráfica de barras', 'bar_chart')
    assert clean_queries(['A b', 'a B', '', None, 'usa pie_chart', 'otra'], 'pie_chart') \
        == ['A b', 'otra']
    tools = pd.DataFrame({'name': ['t1'], 'family': ['f'],
                          'description': ['t1: Hace algo.\n\nArgs:\n    x: y']})
    assert catalog(tools) == '- t1: Hace algo.'
    assert index_rows(tools)['query'].isna().all()
    print('tool_queries self-check OK')
