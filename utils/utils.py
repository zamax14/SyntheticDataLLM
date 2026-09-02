import os
import re
import pandas as pd

from docling.document_converter import DocumentConverter
from utils.logger import Logger
from pandas import DataFrame


MarkDowndExtension: str = '.md'
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
