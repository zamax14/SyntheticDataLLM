# ==============================================================================
# Project Name: Synthetic Data Generator
# Script Name: pattern.py
# Authors: Abraham Sánchez, Ulises Moya, Alejandro Zarate  
# Description:
#   This application helps to create synthetic Q&A data format for LLM training
#   models.
# License: MIT
# ==============================================================================

from dataclasses import dataclass


@dataclass
class Pattern:
    """
    A class that defines various regex patterns used for text processing.
    
    This class contains properties that return regex patterns for matching specific
    text formats, such as figure captions, table and figure references, extra spaces,
    markdown image indicators, questions, and answers. These patterns can be used
    in text parsing and validation tasks.
    """

    @property
    def single_figure_caption(self) -> str:
        return r"Figura\s+\d+[\.|\s*]\s+.*\s+<!-- image -->\s*(?!\s*\|)"

    @property
    def table_figure(self) -> str:
        return r"(Tabla\s+\d+[\.|\s+])|(Figura\s+\d+[\.|\s+])"

    @property
    def extra_spaces(self) -> str:
        return r" +"

    @property
    def markdown_image(self) -> str:
        return r"<!-- image -->"

    @property
    def question(self) -> str:
        return r"\**Q:*\**\s+.+"

    @property
    def answer(self) -> str:
        return r"\**A:*\**\s+.+"

    @property
    def markdown_header(self) -> str:
        return r"\#+"

    @property
    def reasoning_section(self) -> str:
        return r"<think>.*?</think>\n*"
