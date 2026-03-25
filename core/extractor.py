# ==============================================================================
# Project Name: Synthetic Data Generator
# Script Name: extractor.py
# Authors: Abraham Sánchez, Ulises Moya, Alejandro Zarate  
# Description:
#   This application helps to create synthetic Q&A data format for LLM training
#   models.
# License: MIT
# ==============================================================================

import re
import polars as pl

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from tqdm import tqdm
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter
)
from core.paragraph import Paragraph
from core.llm import LLM, Generationtype, LanguageType
from core.taxonomy import (
    Taxonomy,
    QandA,
    TaxonomyDocument,
    Example,
    FormatResult
)
from utils.logger import Logger
from core.pattern import Pattern
from typing import Any


type QandAList = list[str]
type ContextQandAResponses = tuple[list[str], QandAList]
type QuestionAnswerList = list[list[str], list[str]]
type ContextList = list[str]
type SummaryList = list[str]

pattern = Pattern()


@dataclass
class Extractor(ABC):
    text: str = field(init=True, default=None)
    filename: str = field(init=False)
    _filename: str = field(init=False, repr=False)
    llm: LLM = field(default=None)

    @property
    def filename(self) -> str:
        return self._filename

    @filename.setter
    def filename(self, filename: str) -> None:
        self._filename = filename

    @abstractmethod
    def extract(self, language: LanguageType = LanguageType.ES) -> Any: ...

@dataclass
class QandAExtractor(Extractor):
    paragraph: Paragraph = field(default=None)

    def llm_questions_answers(
            self,
            language: LanguageType = LanguageType.ES
        ) -> ContextQandAResponses:
        """
        Genera pares pregunta-respuesta y contextos a partir de una lista de párrafos
        usando un modelo de lenguaje con salida estructurada Pydantic.

        Args:
            language: (LanguageType, optional): El tipo de idioma a utilizar.
                                                Por defecto LanguageType.ES

        Returns:
            Tuple[ContextList, QandAResponses]: Tupla con dos listas:
                - contexts: Lista de contextos generados para cada párrafo.
                - responses: Lista de pares pregunta-respuesta para cada párrafo.
        """
        responses = list()
        contexts = list()
        for p in tqdm(self.paragraph, ascii=True, ncols=75, desc='🤖🔧 Q&A'):
            context = self.llm.generate(
                text=p, gtype=Generationtype.TRANSLATE
            )
            try:
                qa_response = self.llm.generate_qa(text=context)
                qas = [
                    [item.prompt, item.completion]
                    for item in qa_response.pairs
                ]
            except Exception:
                Logger.warning('🟡 No se pudo extraer Q&A del párrafo.')
                continue
            if not qas:
                Logger.warning('🟡 No se pudo extraer Q&A del párrafo.')
                continue
            contexts.append(context)
            responses.append(qas)
        return contexts, responses

    def format(
            self,
            questions_answers: QuestionAnswerList,
            contexts: ContextList
        ) -> FormatResult:
        """
        Format the question-answer pairs and their corresponding contexts into a structured taxonomy.

        Args:
            questions_answers (QuestionAnswerList): A list of question-answer pairs.
            contexts (ContextList): A list of contexts corresponding to the question-answer pairs.

        Returns:
            str: A formatted string representation of the taxonomy.
        """
        examples = list()
        for context, questions_answer in zip(contexts, questions_answers):
            questions, answers = zip(*questions_answer)
            example = Example(
                context=context,
                qa=QandA(
                    questions=questions,
                    answers=answers
                )
            )
            examples.append(example)

        taxonomy = Taxonomy(
            seed_examples=examples,
            document_outline='A description',
            domain='Documentos',
            document=TaxonomyDocument(patterns=[self.filename])
        ).format()
        return taxonomy

    def extract(
            self,
            language: LanguageType = LanguageType.ES
        ) -> FormatResult:
        """
        Extract question-answer pairs and their contexts, and format them into a final output.

        Args:
            language: (LanguageType, optional): The type of language to perform.
                                                Defaults to LanguageType.ES

        Returns:
           str: The final extraction result formatted as a string.
        """
        contexts, responses = self.llm_questions_answers(language=language)
        extraction = self.format(questions_answers=responses, contexts=contexts)
        return extraction


class SummarizeExtractor(Extractor):
    """
    Extract and summarize text from markdown documents using a language model.
    Inherits from the Extractor class.
    """

    def extract(self, language = LanguageType.ES) -> str:
        """
        Extracts and summarizes the text from the markdown content.

        Args:
            language (LanguageType): The language type for the summarization (default is Spanish).

        Returns:
            str: The formatted summary of the extracted content.
        """
        headers = [
            ("##", "Header 2"),
        ]
        md_splitter = MarkdownHeaderTextSplitter(headers)
        md_documents = md_splitter.split_text(self.text)
        content = str()
        header = None
        for document in md_documents:
            if not header:
                header = document.metadata.get('Header 2')
            content += ('\n\n' + document.page_content)
        paragraph = Paragraph(text=content)
        summaries = self.llm_summarize(paragraphs=paragraph, language=language)
        formated = self.format(summaries=summaries, header=header)
        return formated

    def llm_summarize(
        self,
        paragraphs: Paragraph,
        language: LanguageType = LanguageType.ES
    ) -> SummaryList:
        """
        Generates summaries for the provided paragraphs using a language model.

        Args:
            paragraphs (Paragraph): The paragraphs to be summarized.
            language (LanguageType): The language type for the summarization (default is Spanish).

        Returns:
            SummaryList: A list of generated summaries for the paragraphs.
        """
        summaries = list()
        for p in tqdm(paragraphs, ascii=True, ncols=75, desc='✍ 🤖 Summary'):
            summary = self.llm.generate(
                text=p, gtype=Generationtype.SUMMARIZE
            )
            summaries.append(summary)
        return summaries

    def format(self, summaries: SummaryList, header: str | None) -> str:
        """
        Formats the generated summaries into a markdown string.

        Args:
            summaries (SummaryList): The list of summaries to format.
            header (str | None): An optional header to include in the formatted output.

        Returns:
            str: The formatted markdown string containing the summaries.
        """
        buffer = str()
        for summary in summaries:
            summary = re.sub(pattern.markdown_header, '###', summary)
            buffer += (summary + '\n\n')
        if header:
            header = '## ' + header + '\n\n'
            buffer = header + buffer
        return buffer


class ReasoningExtractor(Extractor):
    """
    A class to extract reasoning from a dataset using a language model.
    Inherits from the Extractor class.
    """

    def extract(self, language = LanguageType.ES) -> pl.DataFrame:
        """
        Extracts reasoning from a CSV file and returns a DataFrame with the results.

        Args:
            language (LanguageType): The language type for the reasoning generation. 
                                      Defaults to Spanish (LanguageType.ES).

        Returns:
            pl.DataFrame: A DataFrame containing the expected answers, problems, 
                          and generated solutions.
        """
        df = pl.read_csv(self.filename, encoding='latin')

        expected_columns = 3
        if df.width != expected_columns:
            raise ValueError(f"Expected {expected_columns} columns, got {df.width}.")

        responses = list()

        for instruction, answer, source in tqdm(
            df.iter_rows(), ascii=True, ncols=75,
            desc='🧠  Generating', total=df.height
        ):

            try:
                response = self.llm.generate_reasoning(
                    question=instruction,
                    answer=answer,
                    source=source
                )
            except Exception as e:
                Logger.warning(f'🟡  Skipping Row: {str(e)}')
                continue

            responses.append([answer, instruction, response])

        return pl.DataFrame(
            responses,
            schema=['expected_answer', 'problem', 'generated_solution'],
            orient='row'
        )

    def clean_response(self, text: str) -> str:
        """
        Cleans the raw output from the language model by removing the reasoning section 
        pattern and stripping extra whitespace.

        Args:
            text (str): The raw response generated by the language model.

        Returns:
            str: A cleaned version of the response, with the reasoning section removed.
        """
        return re.sub(pattern.reasoning_section, '', text, flags=re.DOTALL).strip()
