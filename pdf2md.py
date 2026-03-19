import os

from jsonargparse import CLI
from dataclasses import dataclass
from enum import Enum
from tqdm import tqdm
from docling.document_converter import DocumentConverter
from utils.logger import Logger
from utils.utils import to_markdown, save_markdown


class DocType(Enum):
    PDF = ['.pdf', '.PDF']


@dataclass
class PdfToMd:
    """
    A class to convert PDF files to Markdown format.
    """
    
    def convert(
        self,
        data_path: str,
        output_path: str
    ) -> None:
        """
        Convert PDF files in the specified input directory to Markdown files
        and save them in the specified output directory.

        Args:
            data_path (str): The path to the directory containing PDF files.
            output_path (str): The path to the directory where Markdown files will be saved.
        """
        count = 0
        Logger.info('⚪ Processing PDF converter ...')
        converter = DocumentConverter()
        for root, _, files in os.walk(data_path):
            for filename in tqdm(files, ascii=True, ncols=75, desc='⚙️ Converting'):
                name, ext = os.path.splitext(filename)

                if ext in DocType.PDF.value:
                    result = to_markdown(
                        filename=os.path.join(root, filename),
                        converter=converter
                    )
                    save_markdown(
                        file_content=result,
                        filename=name+'.md',
                        output_path=output_path
                    )
                    count += 1
        if count > 0:
            Logger.info(f'🟢 Process done! Processed [{count}] files.')
        else:
            Logger.warning('🟡 No files converted.')



if __name__ == '__main__':
    CLI(PdfToMd)
