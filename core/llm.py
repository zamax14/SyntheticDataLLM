# ==============================================================================
# Project Name: Synthetic Data Generator
# Script Name: aqllm.py
# Authors: Abraham Sánchez, Ulises Moya, Alejandro Zarate  
# Description:
#   This application helps to create synthetic Q&A data format for LLM training
#   models.
# License: MIT
# ==============================================================================
from dataclasses import dataclass
from enum import Enum, auto
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from core.schemas import QandAResponse, QuestionList, VariantList
from utils.templates import (
    create_q_and_a_prompt_es,
    create_translate_prompt_es,
    create_summarize_prompt_es,
    create_reasoning_prompt_es,
    create_rag_questions_prompt_es,
    create_rag_variants_prompt_es
)


class Generationtype(Enum):
    QUESTION_ANSWER = auto()
    TRANSLATE = auto()
    SUMMARIZE = auto()
    REASONING = auto()


class LanguageType(Enum):
    ES = auto()


@dataclass
class LLM:
    """
    Clase para manejar la generación de pares pregunta-respuesta, traducciones
    y resúmenes usando un modelo de lenguaje especificado.

    Attributes:
        model (str): Nombre o identificador del modelo de lenguaje.
    """
    model: str

    def __post_init__(self) -> None:
        self._ollama = ChatOllama(model=self.model)
        self._structured_qa = self._ollama.with_structured_output(QandAResponse)
        self._structured_questions = self._ollama.with_structured_output(QuestionList)
        self._structured_variants = self._ollama.with_structured_output(VariantList)
        self._prompt_fn = {
            Generationtype.TRANSLATE: create_translate_prompt_es,
            Generationtype.SUMMARIZE: create_summarize_prompt_es,
        }

    def generate(
        self,
        text: str,
        gtype: Generationtype = Generationtype.TRANSLATE
    ) -> str:
        """
        Genera una respuesta de texto libre basada en el tipo de generación.

        Args:
            text (str): El texto de entrada a procesar.
            gtype (Generationtype): El tipo de generación (TRANSLATE o SUMMARIZE).

        Returns:
            str: La respuesta generada por el modelo de lenguaje.
        """
        prompt = self._prompt_fn[gtype]()
        chain = prompt | self._ollama | StrOutputParser()
        response = chain.invoke({'text': text})
        return response

    def generate_qa(self, text: str) -> QandAResponse:
        """
        Genera pares de pregunta y respuesta estructurados usando Pydantic.

        Args:
            text (str): El párrafo de entrada para generar Q&A.

        Returns:
            QandAResponse: Objeto Pydantic con la lista de pares pregunta-respuesta.
        """
        prompt = create_q_and_a_prompt_es()
        chain = prompt | self._structured_qa
        response = chain.invoke({'text': text})
        return response

    def generate_reasoning(
        self,
        question: str,
        answer: str,
        source: str
    ) -> str:
        """
        Genera una respuesta de razonamiento a partir de pregunta, respuesta y fuente.

        Args:
            question (str): La pregunta a razonar.
            answer (str): La respuesta esperada.
            source (str): La fuente de información.

        Returns:
            str: La respuesta de razonamiento generada por el modelo.
        """
        prompt = create_reasoning_prompt_es()
        chain = prompt | self._ollama | StrOutputParser()
        response = chain.invoke(
            {
                'question': question,
                'answer': answer,
                'source': source
            }
        )
        return response

    def generate_questions(
        self,
        text: str,
        num_questions: int = 3
    ) -> QuestionList:
        """
        Genera preguntas de validación RAG a partir de un chunk de texto.

        Args:
            text (str): El chunk de texto para generar preguntas.
            num_questions (int): Número de preguntas a generar.

        Returns:
            QuestionList: Objeto Pydantic con la lista de preguntas generadas.
        """
        prompt = create_rag_questions_prompt_es(num_questions)
        chain = prompt | self._structured_questions
        response = chain.invoke({'text': text})
        return response

    def generate_variants(
        self,
        question: str,
        num_variants: int = 2
    ) -> VariantList:
        """
        Genera variantes (reescrituras semánticas) de una pregunta.

        Args:
            question (str): La pregunta original a reescribir.
            num_variants (int): Número de variantes a generar.

        Returns:
            VariantList: Objeto Pydantic con la lista de variantes generadas.
        """
        prompt = create_rag_variants_prompt_es(num_variants)
        chain = prompt | self._structured_variants
        response = chain.invoke({'question': question})
        return response
