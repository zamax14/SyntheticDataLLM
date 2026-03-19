# ==============================================================================
# Project Name: Synthetic Data Generator
# Script Name: taxonomy.py
# Authors: Abraham Sánchez, Ulises Moya, Alejandro Zarate  
# Description:
#   This application helps to create synthetic Q&A data format for LLM training
#   models.
# License: MIT
# ==============================================================================


from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


type FormatResult = dict[str, str | int | list[str]]
type Questions = list[str]
type Answers = list[str]

@dataclass
class Format(ABC):
    @abstractmethod
    def format(self) -> Any: ...


@dataclass
class QandA(Format):
    questions: Questions
    answers: Answers

    def format(self) -> FormatResult:
        values = list()
        for question, answer in zip(self.questions, self.answers):
           values.append({
              'question': question,
              'answer': answer
           })
        return values


@dataclass
class Example(Format):
    context: str
    qa: QandA

    def format(self) -> FormatResult:
        value = {
           'context': f"""{self.context}""",
           'questions_and_answers': self.qa.format()
        }
        return value


@dataclass
class TaxonomyDocument(Format):
    patterns: list[str]
    commit: str = field(default=None)
    repo: str = field(default=None)

    def format(self) -> FormatResult:
        value = {
           'repo': self.repo,
           'commit': None,
           'patterns': self.patterns
        }
        return value


@dataclass
class Taxonomy(Format):
    seed_examples: list[Example]
    document_outline: str
    document: TaxonomyDocument
    domain: str
    created_by: str = field(init=False, default='zamax14')
    version: int = field(init=False, default=3)

    def format(self) -> FormatResult:
        value = {
           'created_by': self.created_by,
           'version': int(self.version),
           'domain': self.domain,
           'seed_examples': [
              example.format() for example in self.seed_examples
           ],
           'document_outline': self.document_outline,
           'document': self.document.format()
        }
        return value
