# ==============================================================================
# Project Name: Synthetic Data Generator
# Script Name: embeddings_pipeline.py
# Description:
#   Generates (query, answer, hard_negative) triplets for embedding model
#   fine-tuning using distilabel's GenerateSentencePair task. Replaces the
#   old translate+free-text-Q&A+regex pipeline for this use case: the LLM's
#   structured output is enforced by the schema distilabel builds internally
#   (no manual JSON parsing), and the prompt explicitly grounds the query in
#   the anchor's specific facts while producing a hard negative alongside it.
# License: MIT
# ==============================================================================

from functools import partial

from distilabel.models import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import GenerateSentencePair

DEFAULT_CONTEXT_ES = (
    'Los fragmentos de "ancla" provienen de documentos oficiales del IIEG '
    '(Instituto de Información Estadística y Geográfica de Jalisco), en español. '
    'La "consulta" (query) generada debe reflejar cómo una persona real buscaría '
    'este contenido en un buscador: debe anclarse en cifras, nombres de '
    'municipios, indicadores o entidades específicas presentes en el ancla, '
    'nunca una pregunta genérica que cualquier documento del mismo tema '
    'podría responder igual de bien. Varía cuánto se parecen las palabras de '
    'la consulta a las del ancla: a veces reutiliza términos exactos, a veces '
    'parafrasea. Responde siempre en español.'
)


class NoThinkOpenAILLM(OpenAILLM):
    """
    OpenAILLM that disables the model's reasoning/thinking mode.

    Needed for Ollama-served reasoning models (e.g. qwen3.6): by default they
    return the answer in the `reasoning` field and leave `content` EMPTY, so
    every generation would be dropped. `reasoning_effort` is injected at the
    OpenAI-SDK level (`extra_body`, a documented client kwarg) instead of
    through distilabel's `generation_kwargs`, which validates its signature and
    would reject an unknown key.
    """

    def load(self) -> None:
        super().load()
        for client in (self._client, self._aclient):
            client.chat.completions.create = partial(
                client.chat.completions.create,
                extra_body={'reasoning_effort': 'none'},
            )


def build_pipeline(
    anchors: list[str],
    model_name: str = 'gpt-4o-mini',
    context: str = DEFAULT_CONTEXT_ES,
    temperature: float = 0.7,
    base_url: str | None = None,
    api_key: str | None = None,
    disable_thinking: bool = False,
) -> Pipeline:
    """
    Builds a distilabel pipeline that turns anchor passages into
    (query, positive, hard_negative) triplets for embedding fine-tuning.

    Args:
        anchors: Cleaned paragraph texts to feed the pipeline. They are passed
                 to `LoadDataFromDicts` at construction time: `data` is not a
                 runtime parameter, so injecting it through `run(parameters=)`
                 leaves the step empty and the run dies on an empty column list.
        model_name: Model id used for generation (OpenAI id, or the Ollama tag
                    when `base_url` points at an Ollama server).
        context: Domain context injected into the generation prompt, used to
                 ground queries in the anchor's specific facts/entities.
        temperature: Sampling temperature for the LLM.
        base_url: OpenAI-compatible endpoint. Point it at Ollama
                  (http://localhost:11434/v1) to generate locally; leave unset
                  for the real OpenAI API.
        api_key: API key. Ollama ignores its value but the client requires one.
        disable_thinking: Required for reasoning models served by Ollama —
                          see `NoThinkOpenAILLM`.

    Returns:
        An unrun distilabel Pipeline, ready for `pipeline.run()`.
    """
    llm_cls = NoThinkOpenAILLM if disable_thinking else OpenAILLM
    llm_kwargs = {
        'model': model_name,
        'generation_kwargs': {'temperature': temperature},
    }
    if base_url:
        llm_kwargs['base_url'] = base_url
    if api_key:
        llm_kwargs['api_key'] = api_key

    with Pipeline(name='embeddings-synthetic-data') as pipeline:
        load_data = LoadDataFromDicts(
            name='load_anchors',
            data=[{'anchor': anchor} for anchor in anchors],
        )
        generate_pairs = GenerateSentencePair(
            name='generate_pairs',
            triplet=True,
            hard_negative=True,
            action='query',
            context=context,
            llm=llm_cls(**llm_kwargs),
            use_default_structured_output=True,
        )
        load_data >> generate_pairs
    return pipeline


def generate_triplets(
    anchors: list[str],
    sources: list[str],
    model_name: str = 'gpt-4o-mini',
    context: str = DEFAULT_CONTEXT_ES,
    base_url: str | None = None,
    api_key: str | None = None,
    disable_thinking: bool = False,
) -> list[dict[str, str]]:
    """
    Runs the pipeline over a list of anchor paragraphs and returns training rows.

    Args:
        anchors: Cleaned paragraph texts to use as the anchor/answer.
        sources: Source filename per anchor (same length/order as `anchors`),
                 kept for traceability in the output dataset.
        model_name: Model id used for generation.
        context: Domain context injected into the generation prompt.
        base_url: OpenAI-compatible endpoint (e.g. an Ollama server).
        api_key: API key for that endpoint.
        disable_thinking: Disable the model's reasoning mode (Ollama).

    Returns:
        A list of {'query', 'answer', 'hard_negative', 'source_file'} dicts.
        Anchors for which generation failed (empty positive/negative) are
        skipped. Rows are matched back to their source by anchor text, so
        two identical paragraphs from different files collapse to one
        source (acceptable: duplicate paragraphs are rare and low-value).
    """
    if not anchors:
        return []

    source_by_anchor = dict(zip(anchors, sources))
    pipeline = build_pipeline(
        anchors=anchors,
        model_name=model_name,
        context=context,
        base_url=base_url,
        api_key=api_key,
        disable_thinking=disable_thinking,
    )
    distiset = pipeline.run(use_cache=False)
    generated = distiset['default']['train']

    rows = []
    for row in generated:
        if not row.get('positive') or not row.get('negative'):
            continue
        rows.append({
            'query': row['positive'],
            'answer': row['anchor'],
            'hard_negative': row['negative'],
            'source_file': source_by_anchor.get(row['anchor'], ''),
        })
    return rows
