# ==============================================================================
# Project Name: Synthetic Data Generator
# Script Name: ragval.py
# Authors: Abraham Sánchez, Ulises Moya, Alejandro Zarate
# Description:
#   CLI para generar datasets de validación de sistemas RAG.
#   Convierte documentos a Markdown, chunkea por headers y genera preguntas
#   con variantes para evaluar la recuperación de chunks.
# License: MIT
# ==============================================================================

import os
import time

import pandas as pd

from jsonargparse import CLI
from dataclasses import dataclass
from docling.document_converter import DocumentConverter
from tqdm import tqdm
from core.chunker import HeaderChunker
from core.llm import LLM
from utils.logger import Logger
from utils.utils import (
    read_data,
    to_markdown,
    save_csv,
    MarkDowndExtension,
    AllowedFileExtensions
)


@dataclass
class RAGValidation:
    """Clase para generar datasets de validación de sistemas RAG."""

    def generate(
        self,
        data_path: str,
        output_path: str,
        model_name: str = 'phi4',
        num_questions: int = 3,
        num_variants: int = 2,
        convert: bool = False,
        min_tokens: int = 50
    ) -> None:
        """
        Genera un dataset de validación RAG a partir de documentos.

        Convierte documentos a Markdown (opcional), chunkea por headers de
        primer nivel (#), genera preguntas por chunk con variantes semánticas
        y exporta un CSV consolidado.

        Args:
            data_path (str): Directorio con los documentos de entrada (.md o documentos a convertir).
            output_path (str): Directorio donde se guardará el CSV de salida.
            model_name (str): Nombre del modelo Ollama a utilizar.
            num_questions (int): Número de preguntas a generar por chunk.
            num_variants (int): Número de variantes por pregunta.
            convert (bool): Si es True, convierte los documentos a Markdown antes de procesar.
            min_tokens (int): Número mínimo de caracteres para considerar un chunk válido.
        """
        Logger.info('🚀 Generando dataset de validación RAG ...')
        llm = LLM(model=model_name)

        md_path = data_path
        if convert:
            md_path = os.path.join(output_path, 'md')
            self._convert_documents(data_path=data_path, md_output=md_path)

        md_files = [
            (root, f) for root, _, files in os.walk(md_path)
            for f in files if os.path.splitext(f)[1] == MarkDowndExtension
        ]
        Logger.info(f'📄 Encontrados {len(md_files)} archivos markdown en {md_path}')

        if not md_files:
            Logger.error('No se encontraron archivos markdown. Verifica data_path o usa --convert true.')
            return

        rows: list[dict[str, str]] = []
        row_id = 1

        for idx, (root, filename) in enumerate(
            tqdm(md_files, ascii=True, ncols=75, desc='⚪ Procesando docs'), 1
        ):
            Logger.info(f'[{idx}/{len(md_files)}] Procesando: {filename}')
            t0 = time.time()

            content = read_data(filename=os.path.join(root, filename))
            if content is None:
                continue

            chunker = HeaderChunker(text=content, min_tokens=min_tokens)
            chunks = chunker.chunk()
            Logger.info(f'  📦 {len(chunks)} chunks extraídos de {filename}')

            for chunk in tqdm(chunks, ascii=True, ncols=75, desc='  📝 Chunks', leave=False):
                try:
                    question_list = llm.generate_questions(
                        text=chunk.content,
                        num_questions=num_questions
                    )
                except Exception as e:
                    Logger.error(f'Error generando preguntas para chunk {chunk.chunk_id}: {e}')
                    continue

                for q_item in question_list.questions:
                    rows.append({
                        'id': row_id,
                        'pregunta': q_item.question,
                        'chunk_id': chunk.chunk_id,
                        'chunk_content': chunk.content,
                        'documento': filename
                    })
                    row_id += 1

                    try:
                        variant_list = llm.generate_variants(
                            question=q_item.question,
                            num_variants=num_variants
                        )
                    except Exception as e:
                        Logger.error(f'Error generando variantes: {e}')
                        continue

                    for variant in variant_list.variants:
                        rows.append({
                            'id': row_id,
                            'pregunta': variant,
                            'chunk_id': chunk.chunk_id,
                            'chunk_content': chunk.content,
                            'documento': filename
                        })
                        row_id += 1

            Logger.info(f'  ✅ {filename} procesado en {time.time() - t0:.1f}s')

        if rows:
            df = pd.DataFrame(rows)
            os.makedirs(output_path, exist_ok=True)
            save_csv(dataframe=df, filename='ragval_dataset', output_path=output_path)
            Logger.info(f'💾 Dataset guardado: {os.path.join(output_path, "ragval_dataset.csv")}')
            Logger.info(f'📊 Total de filas: {len(df)}')
        else:
            Logger.error('No se generaron filas. Verifica los documentos de entrada.')

        Logger.info('🟢 Proceso finalizado!')

    def _convert_documents(self, data_path: str, md_output: str) -> None:
        """
        Convierte documentos del directorio de entrada a Markdown.

        Args:
            data_path (str): Directorio con los documentos originales.
            md_output (str): Directorio donde se guardarán los archivos .md.
        """
        Logger.info(f'📂 Convirtiendo documentos de {data_path} a Markdown ...')
        os.makedirs(md_output, exist_ok=True)
        converter = DocumentConverter()

        doc_files = [
            (root, f) for root, _, files in os.walk(data_path)
            for f in files if os.path.splitext(f)[1].lower() in AllowedFileExtensions
        ]
        Logger.info(f'📄 Encontrados {len(doc_files)} documentos para convertir')

        for root, filename in tqdm(doc_files, ascii=True, ncols=75, desc='📄 Convirtiendo'):
            name, _ = os.path.splitext(filename)
            filepath = os.path.join(root, filename)
            try:
                markdown = to_markdown(filename=filepath, converter=converter)
                md_filepath = os.path.join(md_output, name + MarkDowndExtension)
                with open(md_filepath, 'w', encoding='utf-8') as f:
                    f.write(markdown)
                Logger.info(f'  ✅ Convertido: {filename}')
            except Exception as e:
                Logger.error(f'  ❌ Error convirtiendo {filename}: {e}')


if __name__ == '__main__':
    CLI(RAGValidation)
