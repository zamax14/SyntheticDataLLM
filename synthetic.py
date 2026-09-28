# ==============================================================================
# Project Name: Synthetic Data Generator
# Script Name: synthetic.py
# Authors: Abraham Sánchez, Ulises Moya, Alejandro Zarate
# Description:
#   Generates (query, answer, hard_negative) triplets from Markdown documents
#   for fine-tuning Spanish embedding models (RAG / tool-calling retrieval).
# License: MIT
# ==============================================================================

import hashlib
import os
import pandas as pd

from jsonargparse import CLI
from dataclasses import dataclass
from core import quality_gate
from core.paragraph import Paragraph
from utils.logger import Logger
from utils.utils import read_data, read_csv, MarkDowndExtension


def _write_control(kept: pd.DataFrame, output_path: str, ragval_csv: str | None) -> None:
    """control_test.csv for E1 and, with ragval_csv, ragval_control.csv for E2: the
    control questions plus every ragval chunk as an index-only row."""
    os.makedirs(output_path, exist_ok=True)
    kept[['query', 'answer', 'source_file']].to_csv(
        os.path.join(output_path, 'control_test.csv'), index=False)
    Logger.info(f'🟢 {len(kept)} control queries for {kept.answer.nunique()} test passages')
    if not ragval_csv:
        return
    chunk_id = lambda c: hashlib.sha256(str(c).encode()).hexdigest()[:12]
    index = read_csv(ragval_csv).drop_duplicates('chunk_id')
    control = pd.DataFrame({
        'pregunta': kept['query'], 'chunk_id': kept['answer'].map(chunk_id),
        'chunk_content': kept['answer'], 'documento': kept['source_file'],
    })
    missing = set(control.chunk_id) - set(index.chunk_id)
    assert not missing, f'{len(missing)} test passages are not ragval chunks'
    out = pd.concat([control, index[['chunk_id', 'chunk_content', 'documento']]
                     .assign(pregunta=None)], ignore_index=True)
    out.insert(0, 'id', range(1, len(out) + 1))
    out.to_csv(os.path.join(output_path, 'ragval_control.csv'), index=False)
    Logger.info(f'🟢 E2 control: {len(control)} questions over {out.chunk_id.nunique()} indexed chunks')


@dataclass
class SyntheticData:
    """Class for handling synthetic data operations."""

    def create_embeddings(
        self,
        data_path: str,
        output_path: str,
        model_name: str = 'gpt-4o-mini',
        context: str | None = None,
        min_tokens: int = 200,
        max_new_tokens: int = 512,
        input_batch_size: int = 50,
        store_jsonl: bool = False,
        base_url: str | None = None,
        api_key: str | None = None,
        disable_thinking: bool = False
    ) -> None:
        """
        Create (query, answer, hard_negative) triplets for embedding model
        fine-tuning, grounded in the specific facts of each paragraph.

        Uses distilabel's GenerateSentencePair task: the library owns the
        prompt and the parsing, and each anchor paragraph yields a query
        anchored in its entities/figures plus an LLM-authored hard negative
        in one call.
        Generated pairs go through the protocol's quality gate before being
        written; rejects are kept in rejected_qa.csv for inspection.

        Runs against OpenAI (needs OPENAI_API_KEY exported) or, by setting
        base_url, against any OpenAI-compatible server such as Ollama.

        Args:
            data_path (str): The input directory with markdown files.
            output_path (str): The output directory path.
            model_name (str): Model id (OpenAI id, or the Ollama tag).
            context (str): Domain context injected into the generation prompt.
                            Defaults to the IIEG Spanish context of the pipeline.
            min_tokens (int): Minimum paragraph length to use as an anchor.
            max_new_tokens (int): Output budget per generation; too low truncates
                                  the answer and the row is dropped.
            input_batch_size (int): Anchors dispatched concurrently to the server.
            store_jsonl (bool): Also store a JSONL copy alongside the CSV.
            base_url (str): OpenAI-compatible endpoint. Point it at Ollama
                            (http://localhost:11434/v1) to generate locally.
            api_key (str): API key for that endpoint ('ollama' works for Ollama).
            disable_thinking (bool): Required for Ollama reasoning models such as
                                     qwen3.6, whose answer is otherwise empty.
        """
        Logger.info('🚀 Generating embeddings training data (query, answer, hard_negative) ...')
        # Imported here so the other commands stay usable in environments
        # without distilabel (the miner only needs sentence-transformers).
        from core.embeddings_pipeline import DEFAULT_CONTEXT_ES, generate_triplets
        context = context or DEFAULT_CONTEXT_ES

        anchors, sources = [], []
        for root, _, files in os.walk(data_path):
            for filename in files:
                _, ext = os.path.splitext(filename)
                if ext != MarkDowndExtension:
                    continue
                content = read_data(filename=os.path.join(root, filename))
                paragraph = Paragraph(text=content, min_tokens=min_tokens)
                anchors.extend(quality_gate.clean_passage(p) for p in paragraph)
                sources.extend([filename] * len(paragraph))

        rows = generate_triplets(
            anchors=anchors, sources=sources, model_name=model_name, context=context,
            max_new_tokens=max_new_tokens, input_batch_size=input_batch_size,
            base_url=base_url, api_key=api_key, disable_thinking=disable_thinking
        )
        if not rows:
            Logger.warning('🟡 No triplets generated.')
            return

        os.makedirs(output_path, exist_ok=True)
        df, rejected = quality_gate.apply(pd.DataFrame(rows))
        Logger.info(quality_gate.report(df, rejected))
        if len(rejected):
            rejected.to_csv(os.path.join(output_path, 'rejected_qa.csv'), index=False)
        if df.empty:
            Logger.warning('🟡 Every generated pair was rejected by the quality gate.')
            return
        df.to_csv(os.path.join(output_path, 'embeddings_qa.csv'), index=False)
        if store_jsonl:
            df.to_json(
                os.path.join(output_path, 'embeddings_qa.jsonl'),
                orient='records', lines=True, force_ascii=False
            )
        Logger.info(f'🟢 Generated {len(df)} triplets -> {output_path}')

    def create_control_queries(
        self,
        input_csv: str,
        output_path: str,
        ragval_csv: str | None = None,
        model_name: str = 'gpt-4o-mini',
        context: str | None = None,
        max_new_tokens: int = 512,
        input_batch_size: int = 50,
        base_url: str | None = None,
        api_key: str | None = None,
        disable_thinking: bool = False
    ) -> None:
        """
        Generator-bias control: new queries for the SAME test passages, written
        by another LLM with the same pipeline and quality gate. If a fine-tuned
        model's gain vanishes on them, it learned the training generator's
        style, not the domain.

        Args:
            input_csv (str): Test partition (answer and source_file columns).
            output_path (str): Directory for control_test.csv (E1) and, with
                               ragval_csv, ragval_control.csv (E2).
            ragval_csv (str): Tesis-RAG ragval_dataset.csv; every one of its
                              chunks stays in the E2 index as index-only rows.
            model_name (str): The other generator (Ollama tag or OpenAI id).
            context (str): Domain context; defaults to the pipeline's.
            max_new_tokens (int): Output budget per generation.
            input_batch_size (int): Anchors dispatched concurrently.
            base_url (str): OpenAI-compatible endpoint.
            api_key (str): API key for that endpoint.
            disable_thinking (bool): Required for Ollama reasoning models.
        """
        Logger.info(f'🚀 Generating control queries with {model_name} ...')
        from core.embeddings_pipeline import DEFAULT_CONTEXT_ES, generate_triplets
        test = read_csv(input_csv).drop_duplicates('answer')
        rows = generate_triplets(
            anchors=test['answer'].tolist(), sources=test['source_file'].tolist(),
            model_name=model_name, context=context or DEFAULT_CONTEXT_ES,
            max_new_tokens=max_new_tokens, input_batch_size=input_batch_size,
            base_url=base_url, api_key=api_key, disable_thinking=disable_thinking
        )
        if not rows:
            raise RuntimeError(f'{model_name} returned no parsable query: check '
                               'disable_thinking and max_new_tokens')
        kept, rejected = quality_gate.apply(pd.DataFrame(rows))
        Logger.info(quality_gate.report(kept, rejected))
        _write_control(kept, output_path, ragval_csv)

    def create_paraphrase_queries(
        self,
        input_csv: str,
        output_path: str,
        ragval_csv: str | None = None,
        max_coverage: float = 0.5,
        model_name: str = 'gpt-4o-mini',
        base_url: str | None = None,
        api_key: str | None = None,
        disable_thinking: bool = False,
        temperature: float = 0.7,
        max_new_tokens: int = 1024,
        seed: int | None = 42,
        workers: int = 8
    ) -> None:
        """
        Lexical control: each test question rewritten without the passage's
        wording, kept only if an answerability judge still says the passage
        answers it and at most `max_coverage` of its content words appear in
        the passage. Writes the same files as create_control_queries plus
        paraphrases.csv with every attempt.

        Args:
            input_csv (str): Test partition (query, answer, source_file).
            output_path (str): Directory for the outputs.
            ragval_csv (str): Tesis-RAG ragval_dataset.csv, for the E2 index.
            max_coverage (float): Max share of content words shared with the passage.
            model_name (str): Model id (OpenAI id, or the Ollama tag).
            base_url (str): OpenAI-compatible endpoint.
            api_key (str): API key for that endpoint.
            disable_thinking (bool): Required for Ollama reasoning models.
            temperature (float): Sampling temperature.
            max_new_tokens (int): Output budget per call.
            seed (int): Sampling seed, for reproducibility.
            workers (int): Concurrent calls to the server.
        """
        Logger.info(f'🚀 Paraphrasing test questions with {model_name} ...')
        from core import paraphrase as para
        from core.tool_queries import ToolLLM
        test = read_csv(input_csv).drop_duplicates('answer').reset_index(drop=True)
        llm = ToolLLM(model_name, base_url, api_key, disable_thinking, temperature,
                      max_new_tokens, seed)
        df = para.paraphrase(llm, test, workers)
        os.makedirs(output_path, exist_ok=True)
        df.to_csv(os.path.join(output_path, 'paraphrases.csv'), index=False)
        keep = df['answerable'] & (df['coverage_paraphrase'] <= max_coverage)
        Logger.info(
            f'Coverage (share of question words in the passage), mean: original '
            f'{df.coverage_original.mean():.2f}, paraphrase {df.coverage_paraphrase.mean():.2f}; '
            f'answerable {df.answerable.mean():.0%}; kept {keep.sum()} of {len(df)} '
            f'(coverage <= {max_coverage})'
        )
        _write_control(df[keep].assign(query=df.loc[keep, 'paraphrase']), output_path, ragval_csv)

    def build_tool_training(
        self,
        input_csv: str,
        test_csv: str,
        tools_csv: str,
        output_path: str,
        confusable_tools: dict[str, list[str]] | None = None,
        max_test_similarity: float = 0.8,
        val_fraction: float = 0.1,
        seed: int = 42
    ) -> None:
        """
        Training pairs (request -> agent tool) for E3 fine-tuning, from a
        create_tool_dataset output. Keeps only the rows where the round-trip
        judge agrees, drops requests too close to any test request (TF-IDF
        cosine >= max_test_similarity) and near-duplicates among themselves,
        splits train/val per tool, and adds a sibling tool (same family or
        declared confusion) as hard negative.

        Args:
            input_csv (str): tool_dataset.csv with query, tool, juez columns.
            test_csv (str): Tesis-RAG toolval_dataset.csv (its questions are the test).
            tools_csv (str): Tesis-Agent's data/tools.csv (name, family, description).
            output_path (str): Directory for train.csv and val.csv.
            confusable_tools (dict): Cross-family confusions, as in create_tool_dataset.
            max_test_similarity (float): TF-IDF cosine above which a request is dropped.
            val_fraction (float): Share of each tool's requests held out for validation.
            seed (int): Seed for the split and the negative choice.
        """
        Logger.info('🚀 Building the tool training pairs ...')
        import random
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        from core.tool_queries import siblings

        df = read_csv(input_csv)
        tools = read_csv(tools_csv)
        described = dict(zip(tools['name'], tools['description']))
        q = df[(df['kind'] == 'positive') & (df['juez'] == df['tool'])].drop_duplicates('query')
        test = read_csv(test_csv)['pregunta'].dropna().tolist()

        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(test + q['query'].tolist())
        to_test = cosine_similarity(vec.transform(q['query']), vec.transform(test)).max(axis=1)
        q = q[to_test < max_test_similarity].reset_index(drop=True)
        within = cosine_similarity(vec.transform(q['query']))
        keep = [i for i in range(len(q)) if not (within[i, :i] >= max_test_similarity).any()]
        q = q.iloc[keep].reset_index(drop=True)

        rng = random.Random(seed)
        rows = []
        for tool, group in q.groupby('tool'):
            group = group.sample(frac=1, random_state=seed)
            n_val = max(1, round(len(group) * val_fraction))
            near = siblings(tools, tool, confusable_tools or {}) or [n for n in described if n != tool]
            for i, query in enumerate(group['query']):
                rows.append({'query': query, 'answer': described[tool],
                             'hard_negative_mined': described[rng.choice(near)],
                             'source_file': tool, 'split': 'val' if i < n_val else 'train'})
        pairs = pd.DataFrame(rows)
        os.makedirs(output_path, exist_ok=True)
        for split in ('train', 'val'):
            pairs[pairs['split'] == split].drop(columns='split').to_csv(
                os.path.join(output_path, f'{split}.csv'), index=False)
        Logger.info(
            f'🟢 {len(pairs)} pairs over {pairs.source_file.nunique()} tools '
            f'(judge-approved {len(df[(df.kind == "positive") & (df.juez == df.tool)])}, '
            f'{(to_test >= max_test_similarity).sum()} too close to test): '
            f'{(pairs.split == "train").sum()} train / {(pairs.split == "val").sum()} val -> {output_path}'
        )

    def refilter(self, input_csv: str, output_path: str) -> None:
        """
        Re-apply the anchor cleaning and the quality gate to an already
        generated CSV, so a new gate rule reaches an existing dataset without
        regenerating it.

        Args:
            input_csv (str): CSV produced by `create_embeddings`.
            output_path (str): Directory for the new embeddings_qa.csv and
                               rejected_qa.csv (never the input's directory).
        """
        Logger.info('🚀 Re-applying the quality gate ...')
        df = read_csv(input_csv)
        df['answer'] = df['answer'].map(quality_gate.clean_passage)
        kept, rejected = quality_gate.apply(df)
        Logger.info(quality_gate.report(kept, rejected))
        os.makedirs(output_path, exist_ok=True)
        rejected.to_csv(os.path.join(output_path, 'rejected_qa.csv'), index=False)
        kept.to_csv(os.path.join(output_path, 'embeddings_qa.csv'), index=False)
        Logger.info(f'🟢 {len(kept)} triplets kept -> {output_path}')

    def mine_negatives(
        self,
        input_csv: str,
        output_path: str,
        model_name: str | None = None,
        num_negatives: int = 1
    ) -> None:
        """
        Enrich a (query, answer) CSV with a corpus-grounded hard negative,
        mined by embedding similarity rather than authored by the LLM.

        Args:
            input_csv (str): CSV with 'query' and 'answer' columns (e.g. the
                              output of `create_embeddings`).
            output_path (str): Directory where the enriched CSV is written.
            model_name (str): Baseline SentenceTransformer model for mining.
                              Defaults to the miner's multilingual baseline.
            num_negatives (int): Hard negatives to mine per query.
        """
        Logger.info('🚀 Mining corpus hard negatives ...')
        # Imported here so the generation environment needs no sentence-transformers.
        from core.hard_negative_miner import DEFAULT_MODEL
        from core.hard_negative_miner import mine as mine_hard_negatives_for

        df = read_csv(input_csv)
        corpus = df['answer'].dropna().unique().tolist()
        mined = mine_hard_negatives_for(
            df=df, corpus=corpus, model_name=model_name or DEFAULT_MODEL,
            num_negatives=num_negatives
        )
        os.makedirs(output_path, exist_ok=True)
        out_file = os.path.join(output_path, os.path.basename(input_csv))
        mined.to_csv(out_file, index=False)
        Logger.info(f'🟢 Mined negatives -> {out_file}')

    def create_tool_dataset(
        self,
        tools_csv: str,
        output_path: str,
        near_themes: list[str],
        far_themes: list[str],
        queries_per_tool: int = 20,
        distractors_per_theme: int = 10,
        only_tools: list[str] | None = None,
        confusable_tools: dict[str, list[str]] | None = None,
        model_name: str = 'gpt-4o-mini',
        base_url: str | None = None,
        api_key: str | None = None,
        disable_thinking: bool = False,
        temperature: float = 0.8,
        max_new_tokens: int = 4096,
        seed: int | None = 42,
        workers: int = 8
    ) -> None:
        """
        Build the tool-retrieval evaluation set (E3): requests whose best tool
        is one of the agent's MCP tools, plus distractor tools that only sit in
        the index. A round-trip judge (the same LLM picking the best tool
        for each request) marks disagreements in `revisar`; the `valida`
        column is left for manual review and `export_ragval` drops the rows
        marked 'n'.

        Args:
            tools_csv (str): Tesis-Agent's data/tools.csv (name, family, description).
            output_path (str): Directory where tool_dataset.csv is written.
            near_themes (list[str]): Themes close to the agent's work (analytics,
                                     visualization, geo) for hard distractors.
            far_themes (list[str]): Unrelated themes for easy distractors.
            queries_per_tool (int): Requests asked per tool.
            distractors_per_theme (int): Distractor tools asked per theme.
            only_tools (list[str]): Restrict to these tools (smoke test).
            confusable_tools (dict): Cross-family confusions, tool -> tools a
                                     request could be mistaken for; the prompt
                                     shows them next to the tool's own family.
            model_name (str): Model id (OpenAI id, or the Ollama tag).
            base_url (str): OpenAI-compatible endpoint.
            api_key (str): API key for that endpoint.
            disable_thinking (bool): Required for Ollama reasoning models.
            temperature (float): Sampling temperature.
            max_new_tokens (int): Output budget per call.
            seed (int): Sampling seed, for reproducibility.
            workers (int): Concurrent calls to the server.
        """
        Logger.info('🚀 Generating the tool-retrieval set ...')
        from core import tool_queries
        tools = read_csv(tools_csv)
        if only_tools:
            tools = tools[tools['name'].isin(only_tools)].reset_index(drop=True)
        llm = tool_queries.ToolLLM(model_name, base_url, api_key, disable_thinking,
                                   temperature, max_new_tokens, seed)

        queries = tool_queries.generate_queries(llm, tools, queries_per_tool, workers,
                                                confusable_tools)
        distractors = tool_queries.generate_distractors(
            llm, tools, {'near': near_themes, 'far': far_themes}, distractors_per_theme, workers
        )
        if len(queries):
            queries['juez'] = tool_queries.judge(llm, tools, queries, workers)
            queries['revisar'] = (queries['juez'] != queries['tool']).map({True: 'si', False: ''})
            Logger.info(f'Round-trip judge disagrees on {(queries.revisar == "si").sum()} '
                        f'of {len(queries)} requests (marked revisar = si)')
        df = pd.concat([queries, tool_queries.index_rows(tools), distractors], ignore_index=True)
        df['valida'] = ''

        os.makedirs(output_path, exist_ok=True)
        out_file = os.path.join(output_path, 'tool_dataset.csv')
        df.to_csv(out_file, index=False)
        per_tool = queries.groupby('tool').size() if len(queries) else pd.Series(dtype=int)
        Logger.info(
            f'🟢 {len(queries)} requests over {len(tools)} tools '
            f'(min {per_tool.min() if len(per_tool) else 0} per tool), '
            f'{len(distractors)} distractors '
            f'({(distractors.kind == "distractor_near").sum() if len(distractors) else 0} near) '
            f'-> {out_file}'
        )
        missing = set(tools['name']) - set(per_tool.index)
        if missing:
            Logger.warning(f'🟡 Tools without requests: {sorted(missing)}')

    def restyle_distractors(
        self,
        input_csv: str,
        tools_csv: str,
        output_path: str,
        batch_size: int = 10,
        model_name: str = 'gpt-4o-mini',
        base_url: str | None = None,
        api_key: str | None = None,
        disable_thinking: bool = False,
        temperature: float = 0.2,
        max_new_tokens: int = 4096,
        seed: int | None = 42,
        workers: int = 8
    ) -> None:
        """
        Rewrite the kept distractor tools of a reviewed tool dataset in the
        agent's docstring style (short first line, short Args, plain ASCII),
        so the index compares function and not wording. The original text is
        kept in `answer_original`; requests and agent tools are untouched.

        Args:
            input_csv (str): Reviewed tool_dataset.csv (with the `valida` column).
            tools_csv (str): Tesis-Agent's data/tools.csv, the style reference.
            output_path (str): Directory where the restyled tool_dataset.csv goes.
            batch_size (int): Distractors rewritten per call.
            model_name (str): Model id (OpenAI id, or the Ollama tag).
            base_url (str): OpenAI-compatible endpoint.
            api_key (str): API key for that endpoint.
            disable_thinking (bool): Required for Ollama reasoning models.
            temperature (float): Sampling temperature (low: this is a rewrite).
            max_new_tokens (int): Output budget per call.
            seed (int): Sampling seed, for reproducibility.
            workers (int): Concurrent calls to the server.
        """
        Logger.info('🚀 Restyling distractor tools ...')
        from core import tool_queries
        df = pd.read_csv(input_csv, keep_default_na=False)
        tools = read_csv(tools_csv)
        llm = tool_queries.ToolLLM(model_name, base_url, api_key, disable_thinking,
                                   temperature, max_new_tokens, seed)

        target = df['kind'].str.startswith('distractor') & (df['valida'].str.lower() != 'n')
        rewritten = tool_queries.restyle(llm, tools, df[target], batch_size, workers)
        failed = rewritten.eq('')
        if failed.any():
            Logger.warning(f'🟡 {failed.sum()} distractors came back empty, kept as they were: '
                           f'{sorted(df.loc[failed[failed].index, "tool"])}')
        df['answer_original'] = ''
        done = rewritten[~failed].index
        df.loc[done, 'answer_original'] = df.loc[done, 'answer']
        df.loc[done, 'answer'] = rewritten[~failed]

        words = df['answer'].map(tool_queries.first_line_words)
        agent = tools['description'].map(tool_queries.first_line_words)
        restyled = words[target]
        Logger.info(
            f'First-line words, median [max]: agent {agent.median():.0f} [{agent.max()}], '
            f'distractors {restyled.median():.0f} [{restyled.max()}]; '
            f'{(restyled > agent.max()).sum()} distractors longer than any agent tool; '
            f'non-ASCII chars left: {df.loc[target, "answer"].map(lambda t: sum(ord(c) > 127 for c in t)).sum()}'
        )
        os.makedirs(output_path, exist_ok=True)
        out_file = os.path.join(output_path, 'tool_dataset.csv')
        df.to_csv(out_file, index=False)
        Logger.info(f'🟢 {len(done)} of {target.sum()} distractors restyled -> {out_file}')

    def export_ragval(
        self,
        input_csv: str,
        output_path: str,
        query_col: str = 'query',
        answer_col: str = 'answer',
        source_col: str = 'source_file'
    ) -> None:
        """
        Derive the context-retrieval evaluation set from a generated CSV.

        Emits the schema Tesis-RAG expects (id, pregunta, chunk_id,
        chunk_content, documento), so a single generation feeds both the
        embedding trainer and the RAG evaluator, and the passage's source
        document travels with every row — which is what allows the
        document-grouped split that removes the train/test leakage.
        Rows marked 'n' in an optional `valida` column (manual review) are
        dropped; rows without a query are exported too, as index-only chunks.

        Args:
            input_csv (str): CSV produced by `create_embeddings`.
            output_path (str): Directory where ragval_dataset.csv is written.
            query_col (str): Column holding the query.
            answer_col (str): Column holding the passage.
            source_col (str): Column holding the source document filename.
        """
        Logger.info('🚀 Exporting RAG evaluation set ...')
        df = read_csv(input_csv)
        if 'valida' in df:
            rejected = df['valida'].astype(str).str.strip().str.lower().eq('n')
            Logger.info(f'Manual review: {rejected.sum()} rows marked invalid, dropped')
            df = df[~rejected].reset_index(drop=True)
        out = pd.DataFrame({
            'id': range(1, len(df) + 1),
            'pregunta': df[query_col],
            # Same hash as Tesis-RAG/src/data/loader.py, so chunk ids line up.
            'chunk_id': df[answer_col].map(
                lambda c: hashlib.sha256(str(c).encode()).hexdigest()[:12]
            ),
            'chunk_content': df[answer_col],
            'documento': df[source_col],
        })
        os.makedirs(output_path, exist_ok=True)
        out_file = os.path.join(output_path, 'ragval_dataset.csv')
        out.to_csv(out_file, index=False)
        Logger.info(
            f'🟢 {out.pregunta.notna().sum()} questions over {out.chunk_id.nunique()} chunks '
            f'from {out.documento.nunique()} documents -> {out_file}'
        )


if __name__ == '__main__':
    CLI(SyntheticData)
