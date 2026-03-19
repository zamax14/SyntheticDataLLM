# ==============================================================================
# Project Name: Synthetic Data Generator
# Script Name: synthetic.py
# Authors: Abraham Sánchez, Ulises Moya, Alejandro Zarate  
# Description:
#   This application helps to create synthetic Q&A data format for LLM training
#   models.
# License: MIT
# ==============================================================================

import os
import yaml
import time

from jsonargparse import CLI
from dataclasses import dataclass
from tqdm import tqdm
from core.paragraph import Paragraph
from core.llm import LLM
from core.extractor import (
    QandAExtractor,
    SummarizeExtractor,
    ReasoningExtractor
)
from core.llm import LanguageType
from utils.logger import Logger
from utils.utils import (
    save_data,
    read_data,
    save_markdown,
    MarkDowndExtension,
    YamlExtension,
    CSVExtension
)


@dataclass
class SyntheticData:
    """Class for handling synthetic data operations."""

    def create(
        self,
        data_path: str,
        output_path: str,
        model_name: str = 'phi4',
        store_csv: bool = False,
        store_jsonl: bool = False,
        language: LanguageType = LanguageType.ES
    ) -> None:
        """
        Create synthetic Q&A data format for LLM training models.\
        The format of the file is based on [InstructLab](https://instructlab.ai/).

        Args:
            data_path (str): The input directory data path.
            output_path (str): The output directory path.
            model_name (str): The ollama model name.
            store_csv (bool): The flag that indicates to store the data as CSV.
            store_jsonl (bool): The flag that indicates to store the data as JSONL.
            language: (LanguageType, optional): The type of language to perform.
                                                Defaults to LanguageType.ES
        """
        Logger.info('🚀 Processing Synthetic data ...')
        llm = LLM(model=model_name)
        md_files = [
            (root, f) for root, _, files in os.walk(data_path)
            for f in files if os.path.splitext(f)[1] == MarkDowndExtension
        ]
        Logger.info(f'📄 Found {len(md_files)} markdown files in {data_path}')
        for idx, (root, filename) in enumerate(tqdm(md_files, ascii=True, ncols=75, desc='⚪ Generating Q&A'), 1):
                name, ext = os.path.splitext(filename)
                Logger.info(f'[{idx}/{len(md_files)}] Processing: {filename}')
                t0 = time.time()
                content = read_data(filename=os.path.join(root, filename))
                if content is None:
                    continue
                extractor = QandAExtractor(
                    text=content, llm=llm,
                    paragraph=Paragraph(text=content)
                )
                extractor.filename = filename
                extraction = extractor.extract(language=language)
                save_data(
                    extraction=extraction, filename=name, output_path=output_path,
                    store_csv=store_csv, store_jsonl=store_jsonl
                )
                Logger.info(f'✅ {filename} done in {time.time() - t0:.1f}s')
        Logger.info('🟢 Process done!')

    def create_reasoning(self,
        data_path: str,
        output_path: str,
        model_name: str = 'qwen3:8b',
        language: LanguageType = LanguageType.ES
    ) -> None:
        Logger.info('🚀 Processing Reasoning Synthetic Data ...')
        llm = LLM(model=model_name)
        csv_files = [
            (root, f) for root, _, files in os.walk(data_path)
            for f in files if os.path.splitext(f)[1] == CSVExtension
        ]
        Logger.info(f'📄 Found {len(csv_files)} CSV files in {data_path}')
        for idx, (root, filename) in enumerate(tqdm(csv_files, ascii=True, ncols=75, desc='⚪ Generating Reasoning'), 1):
                name, _ = os.path.splitext(filename)
                Logger.info(f'[{idx}/{len(csv_files)}] Processing: {filename}')
                t0 = time.time()
                extractor = ReasoningExtractor(
                    filename=os.path.join(root, filename),
                    llm=llm
                )
                extraction = extractor.extract(language=language)
                os.makedirs(output_path, exist_ok=True)
                extraction.write_csv(os.path.join(output_path, name + '.csv'))
                Logger.info(f'✅ {filename} done in {time.time() - t0:.1f}s')
        Logger.info('🟢 Process done!')

    def convert(
        self,
        data_path: str,
        output_path: str
    ) -> None:
        """
        Convert the Q&A files into CSV format.

        Args:
            data_path (str): The input directory data path.
            output_path (str): The output directory path.
        """
        Logger.info('🚀 Processing data conversion ...')
        for root, _, files in os.walk(data_path):
            for filename in tqdm(files, ascii=True, ncols=75, desc='📂 Reading files'):
                name, ext = os.path.splitext(filename)
                if ext != YamlExtension:
                    continue
                with open(os.path.join(root, filename), 'r') as file:
                    data = yaml.safe_load(file)
                    save_data(
                        extraction=data, filename=name, output_path=output_path,
                        store_csv=True, store_jsonl=True
                    )
        Logger.info('🟢 Process done!')

    def summarize(
        self,
        data_path: str,
        output_path: str,
        model_name: str = 'phi4',
        language: LanguageType = LanguageType.ES
    ) -> None:
        """
        Summarizes markdown files from a specified directory and saves the summaries to an output directory.

        Args:
            data_path (str): The path to the directory containing markdown files to be summarized.
            output_path (str): The path to the directory where the summarized markdown files will be saved.
            model_name (str): The name of the language model to be used for summarization (default is 'phi4').
            language (LanguageType): The language type for the summarization (default is Spanish).
        """
        Logger.info('🚀 Processing summarization ...')
        llm = LLM(model=model_name)
        md_files = [
            (root, f) for root, _, files in os.walk(data_path)
            for f in files if os.path.splitext(f)[1] == MarkDowndExtension
        ]
        Logger.info(f'📄 Found {len(md_files)} markdown files in {data_path}')
        for idx, (root, filename) in enumerate(tqdm(md_files, ascii=True, ncols=75, desc='📂 Reading files'), 1):
                name, ext = os.path.splitext(filename)
                Logger.info(f'[{idx}/{len(md_files)}] Processing: {filename}')
                t0 = time.time()
                content = read_data(filename=os.path.join(root, filename))
                if content is None:
                    continue
                extractor = SummarizeExtractor(
                    text=content, llm=llm
                )
                extractor.filename = filename
                extraction = extractor.extract(language=language)
                save_markdown(
                    file_content=extraction, filename=name+'.md',
                    output_path=output_path
                )
                Logger.info(f'✅ {filename} done in {time.time() - t0:.1f}s')
        Logger.info('🟢 Process done!')


if __name__ == '__main__':
    CLI(SyntheticData)
