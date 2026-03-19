# ==============================================================================
# Project Name: Synthetic Data Generator
# Script Name: paragraph.py
# Authors: Abraham Sánchez, Ulises Moya, Alejandro Zarate  
# Description:
#   This application helps to create synthetic Q&A data format for LLM training
#   models.
# License: MIT
# ==============================================================================

import re

from dataclasses import dataclass, field
from core.pattern import Pattern
from utils.utils import replace_pattern

INIT_TABLE_INDICATOR = '|'
INIT_TOPIC_INDICATOR = '##'
TABLE_CAPTION_INDICATOR = 'Table'

type Paragraphs = list[str]

pattern = Pattern()


@dataclass
class Paragraph:
    """
    A class representing a paragraph, which processes and stores paragraph data
    after applying certain patterns for cleaning and modifying the content.
    """
    text: str
    parts: list[str] = field(init=False, default_factory=list)
    min_tokens: int = field(default=200)

    def __post_init__(self) -> None:
        """
        Post-initialization method that processes the text by removing unwanted patterns
        and splitting it into valid paragraph parts.
        """
        text = self._remove_pattern(text=self.text, pattern=pattern.single_figure_caption)
        text = replace_pattern(text=text, by='Tabla: ', pattern=pattern.table_figure)
        text = replace_pattern(text=text, by=' ', pattern=pattern.extra_spaces)
        text = replace_pattern(text=text, by='', pattern=pattern.markdown_image)
        self.parts = self._get_paragraphs(text=text)

    def _remove_pattern(self, text: str, pattern: str) -> str:
        """
        Removes all occurrences of a given pattern from the text.

        Args:
            text: The input text to process.
            pattern: The regular expression pattern to remove from the text.

        Returns:
            A new text with the pattern removed.
        """
        matches = re.finditer(pattern, text)
        pos_to_remove = [match.span() for match in matches]
        last = 0
        result = list()
        for start, end in pos_to_remove:
            result.append(text[last:start])
            last = end
        result.append(text[last:])
        return ''.join(result)

    def _get_paragraphs(self, text: str) -> Paragraphs:
        """
        Splits the text into paragraphs and processes each to filter out invalid ones
        and handle special cases (e.g., tables or extra context).

        Args:
            text: The input text to split into paragraphs.

        Returns:
            A list of valid paragraphs.
        """
        extra_context = None
        is_table = False
        paragraphs = [p for p in text.split('\n\n') if p]
        valid_paragraphs = list()

        for parag in paragraphs:

            if extra_context and parag.startswith(INIT_TABLE_INDICATOR):
                parag = extra_context + '\n' + parag
                is_table = True

            if (TABLE_CAPTION_INDICATOR in parag or INIT_TOPIC_INDICATOR in parag) \
                and extra_context is None:
                extra_context = parag
                continue

            if len(parag) > self.min_tokens:
                if is_table:
                    parag = replace_pattern(
                        text=parag, by='Tabla: ',
                        pattern=INIT_TOPIC_INDICATOR
                    )
                valid_paragraphs.append(parag)

            extra_context = None
            is_table = False

        return valid_paragraphs

    def __getitem__(self, index: int) -> str:
        """
        Allows indexing into the paragraph's parts list to get a specific paragraph.

        Args:
            index: The index of the paragraph to retrieve.

        Returns:
            The paragraph at the specified index.
        """
        return self.parts[index]

    def __len__(self) -> int:
        """
        Returns the number of valid paragraphs.

        Returns:
            The length of the parts list.
        """
        return len(self.parts)
