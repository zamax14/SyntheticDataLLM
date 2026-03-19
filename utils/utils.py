import os
import re
import yaml
import json
import pandas as pd

from tqdm import tqdm
from docling.document_converter import DocumentConverter
from utils.logger import Logger
from core.taxonomy import FormatResult
from pandas import DataFrame


type JsonlFormat = dict[str, str]
MarkDowndExtension: str = '.md'
CSVExtension: str = '.csv'
YamlExtension: str = '.yaml'
PDFExtension: str = '.pdf'
DocxExtension: str = '.docx'
AllowedFileExtensions: list[str] = [PDFExtension, DocxExtension]


def read_data(
        filename: str
    ) -> str | None:
    """Reads the content of a file.

    Args:
        filename (str): The name of the file to read.

    Returns:
        str | None: The content of the file as a string, or None if the file cannot be read.

    Raises:
        IOError: If there is an issue opening or writing to the file.
    """
    content = None
    try:
        with open(filename, 'r') as ptr:
            content = ptr.read()
    except IOError as e:
        Logger.error(str(e))
    return content


def read_csv(
        filename: str
    ) -> DataFrame:
    content = pd.read_csv(filename)
    return content


def replace_pattern(text: str, by: str, pattern: str) -> str:
    """
    Replaces all occurrences of a given pattern in the text with a specified string.

    Args:
        text: The input text to process.
        by: The string that will replace the matched pattern.
        pattern: The regular expression pattern to search for in the text.

    Returns:
        A new text with the pattern replaced by the specified string.
    """
    return re.sub(pattern, by, text).strip()


def to_markdown(filename: str, converter: DocumentConverter = None) -> str:
    """
    Convert a file to Markdown format.

    This function takes the path to a PDF file as input, converts it to a 
    Markdown representation using the DocumentConverter class, and returns 
    the resulting Markdown content.

    Args:
        filename (str): The path to the PDF file that needs to be converted.
        converter (DocumentConverter, optional): A reusable converter instance.
            If None, a new one is created. Pass a shared instance to avoid
            reloading GPU models on every call.

    Returns:
        str: The Markdown representation of the PDF content.

    Raises:
        FileNotFoundError: If the specified PDF file does not exist.
        ConversionError: If there is an error during the conversion process.
    """
    if converter is None:
        converter = DocumentConverter()
    result = converter.convert(filename)
    markdown = result.document.export_to_markdown()
    return markdown


def save_markdown(
        file_content: str,
        filename: str,
        output_path: str
    )  -> None:
    """
    Saves the given content to a markdown file.

    Parameters:
        file_content (str): The content to be written to the markdown file.
        filename (str): The name of the file to be created (should include .md extension).
        output_path (str): The directory path where the file will be saved.

    Returns:
        None: This function does not return any value.
    """
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, filename), 'w', encoding='utf-8') as file:
        file.write(file_content)


def save_csv(
    dataframe: DataFrame,
    filename: str,
    output_path: str
) -> None:
    """Saves the provided dataframe to a CSV file.

    Args:
        dataframe (DataFrame): The dataframe to be saved.
        filename (str): The name of the file to save the data as (without extension).
        output_path (str): The directory path where the file will be saved.

    Returns:
        None: This function does not return a value.
    """
    dataframe.to_csv(os.path.join(output_path, filename+'.csv'), index=False)


def save_jsonl(
    completions: JsonlFormat,
    filename: str,
    output_path: str
) -> None:
    """
    Save a list of completions to a JSON Lines (JSONL) file.

    Args:
        completions (JsonlFormat): A list of completions to be saved. 
                                    Each completion should be serializable to JSON.
        filename (str): The name of the file (without extension) to which the completions 
                        will be saved.
        output_path (str): The directory path where the JSONL file will be created.

    Returns:
        None: This function does not return a value. It writes the completions to a file.
    
    Raises:
        IOError: If there is an issue opening or writing to the file.
    """
    try:
        with open(
                file=os.path.join(output_path, filename+'.jsonl'),
                mode='w',
                encoding='utf-8'
            ) as file:
            for completion in completions:
                file.write(json.dumps(completion, ensure_ascii=False)+'\n')
    except IOError as e:
        Logger.error(str(e))


def save_data(
    extraction: FormatResult,
    filename: str,
    output_path: str,
    store_csv: bool,
    store_jsonl: bool
) -> None:
    """Saves the provided data to a YAML file.

    Args:
        extraction (str): The data to be saved.
        filename (str): The name of the file to save the data as (without extension).
        output_path (str): The directory path where the file will be saved.
        store_csv (bool): The flag that indicates to store the data as CSV.
        store_jsonl (bool): The flag that indicates to store the data as JSONL.

    Returns:
        None: This function does not return a value.
    """
    os.makedirs(output_path, exist_ok=True)
    with open(
        file=os.path.join(output_path, filename+'.yaml'),
        mode='w',
        encoding='utf-8'
        ) as file:  
            yaml.dump(extraction, file, allow_unicode=True)
    if store_csv:
        conversion = convert2dataframe(data=extraction, filename=filename)
        save_csv(dataframe=conversion, filename=filename, output_path=output_path)
    if store_jsonl:
        conversion = convert2jsonl(data=extraction, filename=filename)
        save_jsonl(completions=conversion, filename=filename, output_path=output_path)


def convert2dataframe(data: FormatResult, filename: str) -> DataFrame:
    """
    Convert a structured data format into a Pandas DataFrame.

    This function extracts context, questions, and answers from the provided data 
    and organizes them into a DataFrame with specified columns.

    Args:
        data (FormatResult): A dictionary containing the data to be converted, 
                             expected to have a 'seed_examples' key with relevant data.
        filename (str): The name of the file from which the data was read, used for logging errors.

    Returns:
        DataFrame: A Pandas DataFrame containing the context, questions, and answers.

    Raises:
        KeyError: If the expected keys are not found in the input data.
    """
    buffer = list()
    try:
        samples = data['seed_examples']
        for sample in samples:
            context = sample['context']
            questions_and_answers = sample['questions_and_answers']
            for qas in questions_and_answers:
                question = qas['question']
                answer = qas['answer']
                buffer.append([question, answer])
    except KeyError:
        Logger.error(f'🔴 Q&A not found in file {filename}!')
    df = pd.DataFrame(data=buffer, columns=['instruction', 'answer'])
    return df


def convert2jsonl(data: FormatResult, filename: str) -> JsonlFormat:
    """
    Convert a structured data format into a JSON Lines (JSONL) format.

    This function extracts questions and answers from the provided data and 
    organizes them into a list of dictionaries suitable for JSONL output.

    Args:
        data (FormatResult): A dictionary containing the data to be converted, 
                             expected to have a 'seed_examples' key with relevant data.
        filename (str): The name of the file from which the data was read, used for logging errors.

    Returns:
        JsonlFormat: A list of dictionaries, each containing a 'prompt' and 'completion' key.

    Raises:
        KeyError: If the expected keys are not found in the input data.
    """
    try:
        buffer = list()
        samples = data['seed_examples']
        for sample in samples:
            questions_and_answers = sample['questions_and_answers']
            for qas in questions_and_answers:
                question = qas['question']
                answer = qas['answer']
                buffer.append({'prompt': question, 'completion': answer})
        return buffer
    except KeyError:
        Logger.error(f'🔴 Q&A not found in file {filename}!')


def convert_many(
        data_path: str,
        output_path: str,
        allowed_extensions: list[str] = AllowedFileExtensions
    ) -> None:
    """
    Convert multiple files from a specified directory to Markdown format.

    This function traverses the directory specified by `data_path`, 
    looking for files with extensions that are included in the 
    `allowed_extensions` list. For each valid file, it converts 
    the file to Markdown format and saves it to the specified 
    `output_path`.

    Args:
        data_path : str
            The path to the directory containing the files to be converted.
            
        output_path : str
            The path to the directory where the converted Markdown files 
            will be saved.
            
        allowed_extensions : list[str], optional
            A list of file extensions that are allowed for conversion. 
            The default is `AllowedFileExtensions`, which typically includes 
            common document formats like '.pdf' and '.docx'.
    """
    for root, _, files in os.walk(data_path):
        for filename in tqdm(files, ascii=True, ncols=75, desc='📂 Converting'):
            name, ext = os.path.splitext(filename)
            if ext not in allowed_extensions:
                continue
            markdown = to_markdown(filename=os.path.join(root, filename))
            save_markdown(file_content=markdown, filename=name+'.md',
                          output_path=output_path
            )
