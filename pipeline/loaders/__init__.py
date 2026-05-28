from .loader_registry import get_loader, LoaderRegistry
from .pdf_reader import PDFLoader
from .docx_reader import DocxLoader
from .excel_reader import ExcelLoader
from .pptx_reader import PptxLoader
from .text_reader import TextLoader, MarkdownLoader
from .ocr_reader import OcrLoader
from .epub_reader import EpubLoader

__all__ = [
    "get_loader", "LoaderRegistry",
    "PDFLoader", "DocxLoader", "ExcelLoader", "PptxLoader",
    "TextLoader", "MarkdownLoader", "OcrLoader", "EpubLoader",
]
