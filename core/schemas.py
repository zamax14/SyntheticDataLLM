# ==============================================================================
# Project Name: Synthetic Data Generator
# Script Name: schemas.py
# Authors: Abraham Sánchez, Ulises Moya, Alejandro Zarate
# Description:
#   Modelos Pydantic para validar y estructurar las respuestas del LLM.
# License: MIT
# ==============================================================================

from pydantic import BaseModel, Field


class QandAItem(BaseModel):
    """Un par de pregunta y respuesta generado a partir de un párrafo."""

    prompt: str = Field(
        description="La pregunta generada a partir del párrafo."
    )
    completion: str = Field(
        description="La respuesta detallada a la pregunta."
    )


class QandAResponse(BaseModel):
    """Lista de pares pregunta-respuesta generados a partir de un párrafo."""

    pairs: list[QandAItem] = Field(
        description="Lista de entre 1 y 3 pares de pregunta y respuesta."
    )
