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
from langchain_ollama.llms import OllamaLLM
from utils.templates import (
    create_q_and_a_prompt_es,
    create_translate_prompt_es,
    create_summarize_prompt_es,
    create_reasoning_prompt_es
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
    A class to handle the generation of question and answer pairs or translations 
    using a specified language model.

    Attributes:
        model (str): The name or identifier of the language model to be used for generation.
    """
    model: str

    def __post_init__(self) -> None:
        self._ollama = OllamaLLM(model=self.model)
        self._prompt_fn = {
            Generationtype.QUESTION_ANSWER: create_q_and_a_prompt_es,
            Generationtype.TRANSLATE: create_translate_prompt_es,
            Generationtype.SUMMARIZE: create_summarize_prompt_es,
            Generationtype.REASONING: create_reasoning_prompt_es
        }

    def generate(
        self,
        text: str,
        gtype: Generationtype = Generationtype.QUESTION_ANSWER
    ) -> str:
        """
        Genera una respuesta basada en el tipo de generación especificado.

        Args:
            text (str): El texto de entrada a procesar.
            gtype (Generationtype, optional): El tipo de generación a realizar.
                                               Por defecto Generationtype.QUESTION_ANSWER.

        Returns:
            str: La respuesta generada por el modelo de lenguaje.
        """
        prompt = self._prompt_fn[gtype]()
        chain = prompt | self._ollama
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
        chain = prompt | self._ollama
        response = chain.invoke(
            {
                'question': question,
                'answer': answer,
                'source': source
            }
        )
        return response
