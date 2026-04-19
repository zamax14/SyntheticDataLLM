# ==============================================================================
# Project Name: Synthetic Data Generator
# Script Name: chunker.py
# Authors: Abraham Sánchez, Ulises Moya, Alejandro Zarate
# Description:
#   Chunkeador de documentos Markdown por headers de primer nivel (#).
#   Genera chunks con identificador basado en hash del contenido.
# License: MIT
# ==============================================================================

import re
import hashlib

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """
    Representa un chunk de texto extraído de un documento Markdown.

    Attributes:
        content (str): Texto completo del chunk (header + body).
        header (str): Texto del header del chunk.
        chunk_id (str): Identificador único basado en hash SHA-256 truncado a 12 caracteres.
    """
    content: str
    header: str
    chunk_id: str = field(init=False)

    def __post_init__(self) -> None:
        """Genera el chunk_id como hash SHA-256 truncado del contenido."""
        self.chunk_id = hashlib.sha256(self.content.encode('utf-8')).hexdigest()[:12]


@dataclass
class HeaderChunker:
    """
    Divide un documento Markdown en chunks por el nivel de header más alto presente.
    Por ejemplo, si el documento solo tiene '##', chunkea por '##'; si tiene '#', chunkea por '#'.

    Attributes:
        text (str): Contenido completo del documento Markdown.
        min_tokens (int): Número mínimo de caracteres para considerar un chunk válido.
    """
    text: str
    min_tokens: int = field(default=50)

    def _top_level_pattern(self) -> str:
        """
        Detecta el nivel de header más alto (menor número de #) presente en el documento.

        Returns:
            str: Patrón regex que coincide únicamente con ese nivel de header.
        """
        for level in range(1, 7):
            prefix = '#' * level + ' '
            if any(line.startswith(prefix) for line in self.text.split('\n')):
                # Solo ese nivel exacto: no debe ir seguido de otro '#'
                return r'^' + re.escape(prefix) + r'(?!#)'
        return r'^# (?!#)'

    def chunk(self) -> list[Chunk]:
        """
        Divide el texto en chunks usando el nivel de header más alto como separador.
        Descarta el contenido previo al primer header y chunks con menos de min_tokens caracteres.

        Returns:
            list[Chunk]: Lista de chunks válidos extraídos del documento.
        """
        header_pattern = self._top_level_pattern()
        lines = self.text.split('\n')
        chunks: list[Chunk] = []
        current_header = None
        current_lines: list[str] = []

        for line in lines:
            if re.match(header_pattern, line):
                if current_header is not None:
                    content = '\n'.join(current_lines).strip()
                    if len(content) >= self.min_tokens:
                        chunks.append(Chunk(content=content, header=current_header))
                current_header = line.lstrip('# ').strip()
                current_lines = [line]
            else:
                if current_header is not None:
                    current_lines.append(line)

        if current_header is not None:
            content = '\n'.join(current_lines).strip()
            if len(content) >= self.min_tokens:
                chunks.append(Chunk(content=content, header=current_header))

        return chunks
