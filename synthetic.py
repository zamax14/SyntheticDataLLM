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
                anchors.extend(list(paragraph))
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

        Args:
            input_csv (str): CSV produced by `create_embeddings`.
            output_path (str): Directory where ragval_dataset.csv is written.
            query_col (str): Column holding the query.
            answer_col (str): Column holding the passage.
            source_col (str): Column holding the source document filename.
        """
        Logger.info('🚀 Exporting RAG evaluation set ...')
        df = read_csv(input_csv)
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
            f'🟢 {len(out)} questions over {out.chunk_id.nunique()} chunks '
            f'from {out.documento.nunique()} documents -> {out_file}'
        )


if __name__ == '__main__':
    CLI(SyntheticData)
