# ==============================================================================
# Project Name: Synthetic Data Generator
# Script Name: hard_negative_miner.py
# Description:
#   Mines corpus-wide hard negatives for (query, answer) pairs using a
#   baseline embedding model. Complements the LLM-authored hard negative from
#   embeddings_pipeline.py with a real distractor pulled from the actual
#   document corpus, which tends to be a stronger negative than one an LLM
#   imagines in isolation (see sentence-transformers' mine_hard_negatives).
# License: MIT
# ==============================================================================

import pandas as pd

from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import mine_hard_negatives

DEFAULT_MODEL = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'


def mine(
    df: pd.DataFrame,
    corpus: list[str] | None = None,
    model_name: str = DEFAULT_MODEL,
    num_negatives: int = 1,
    relative_margin: float = 0.05,
) -> pd.DataFrame:
    """
    Mines a corpus-grounded hard negative for each (query, answer) pair.

    Args:
        df: DataFrame with at least 'query' and 'answer' columns.
        corpus: Extra candidate negative texts beyond `df['answer']` (e.g.
                all paragraphs from the source documents, not just the ones
                that became answers). Strongly recommended: it lets negatives
                come from anywhere in the corpus, not just other Q&A rows.
        model_name: SentenceTransformer model used to embed and search.
        num_negatives: Hard negatives to mine per query. Keep at 1 so the
                       merge below stays 1:1; use `mine_hard_negatives`
                       directly with `output_format='n-tuple'` if you want
                       more per row.
        relative_margin: Minimum relative gap between positive and negative
                          similarity, to avoid marking near-duplicates as
                          negatives.

    Returns:
        A copy of `df` with a 'hard_negative_mined' column added (NaN for
        rows where no valid negative was found within the margin).
    """
    model = SentenceTransformer(model_name)
    dataset = Dataset.from_pandas(df[['query', 'answer']], preserve_index=False)
    mined = mine_hard_negatives(
        dataset=dataset,
        model=model,
        anchor_column_name='query',
        positive_column_name='answer',
        corpus=corpus,
        num_negatives=num_negatives,
        relative_margin=relative_margin,
        sampling_strategy='top',
        output_format='triplet',
        use_faiss=False,
        verbose=False,
    )
    mined_df = mined.to_pandas().rename(columns={'negative': 'hard_negative_mined'})
    return df.merge(
        mined_df[['query', 'answer', 'hard_negative_mined']],
        on=['query', 'answer'],
        how='left',
    )
