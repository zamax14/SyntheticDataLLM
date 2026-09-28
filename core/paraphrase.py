# ==============================================================================
# Project Name: Synthetic Data Generator
# Script Name: paraphrase.py
# Authors: Alejandro Zarate
# Description:
#   Lexical control: rewrite test questions so they no longer share the passage's
#   wording. The quality gate requires lexical anchoring, so the generated test
#   questions copy ~80 % of their content words from the passage and BM25 matches
#   the embeddings; paraphrases test whether a model retrieves by meaning.
# License: MIT
# ==============================================================================

import re
import unicodedata

from concurrent.futures import ThreadPoolExecutor

import pandas as pd

PARAPHRASE_PROMPT = """\
Este pasaje de un documento del IIEG responde la pregunta de abajo.

Pasaje:
{passage}

Pregunta original:
{question}

Reescribe la pregunta como la haría un ciudadano que NO ha leído el pasaje:
- pide la MISMA información, de modo que el pasaje la siga respondiendo;
- NO copies las palabras clave del pasaje: usa sinónimos, términos cotidianos o describe
  el concepto (p. ej. "patrones registrados en el IMSS" -> "empresas con trabajadores dados
  de alta en el seguro social");
- no incluyas cifras exactas del pasaje; conserva lugar y periodo solo si son indispensables;
- una sola oración en español de México.

Responde solo con JSON: {{"pregunta": "..."}}"""

JUDGE_PROMPT = """\
Pasaje:
{passage}

Pregunta:
{question}

¿El pasaje contiene la información que pide la pregunta? Responde solo con JSON:
{{"responde": true}} o {{"responde": false}}"""

# ponytail: short closed-class list, the same one as Tesis-RAG's BM25 baseline.
STOPWORDS = set("""
a al ante bajo con contra de del desde durante e el en entre es esta este esto hacia hasta la
las le lo los mas o para pero por que se segun si sin sobre su sus tras un una unas unos y ya
como cual cuales cuando donde cuanto cuantos cuanta cuantas fue fueron ha han hay ser son
""".split())


def content_tokens(text: str) -> set[str]:
    """Lower-cased, accent-free words of 2+ characters, without stopwords."""
    text = unicodedata.normalize('NFKD', str(text).lower()).encode('ascii', 'ignore').decode()
    return {t for t in re.findall(r'\w{2,}', text) if t not in STOPWORDS}


def coverage(question: str, passage: str) -> float:
    """Share of the question's content words that also appear in the passage."""
    q = content_tokens(question)
    return len(q & content_tokens(passage)) / len(q) if q else 1.0


def paraphrase(llm, test: pd.DataFrame, workers: int) -> pd.DataFrame:
    """Paraphrase + answerability judge per (query, answer) row. Adds columns
    paraphrase, coverage_original, coverage_paraphrase and answerable."""

    def one(row) -> dict:
        new = str(llm(PARAPHRASE_PROMPT.format(passage=row.answer, question=row.query))
                  .get('pregunta', '')).strip()
        ok = bool(new) and llm(JUDGE_PROMPT.format(passage=row.answer, question=new)).get('responde') is True
        return {'paraphrase': new, 'coverage_original': coverage(row.query, row.answer),
                'coverage_paraphrase': coverage(new, row.answer) if new else 1.0, 'answerable': ok}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        out = pd.DataFrame(list(pool.map(one, test.itertuples())), index=test.index)
    return test.join(out)


if __name__ == '__main__':
    assert content_tokens('¿Cuántos patrones había en Jalisco?') == {'patrones', 'habia', 'jalisco'}
    assert coverage('patrones en Jalisco', 'En Jalisco había 100 patrones') == 1.0
    assert coverage('empresas con seguro social', 'En Jalisco había 100 patrones') == 0.0

    class Fake:
        def __call__(self, prompt):
            return {'responde': True} if 'contiene la informacion' in prompt or 'contiene la información' in prompt \
                else {'pregunta': 'empresas con seguro social'}
    df = paraphrase(Fake(), pd.DataFrame({'query': ['patrones en Jalisco'],
                                          'answer': ['En Jalisco había 100 patrones']}), 1)
    assert df.loc[0, 'coverage_original'] == 1.0 and df.loc[0, 'coverage_paraphrase'] == 0.0
    assert bool(df.loc[0, 'answerable'])
    print('paraphrase self-check OK')
