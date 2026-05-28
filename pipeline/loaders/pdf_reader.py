from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF
from loguru import logger

from .loader_registry import LoaderRegistry, DocumentContent


@LoaderRegistry.register([".pdf"])
class PDFLoader:
    def load(self, file_path: Path) -> DocumentContent:
        doc = fitz.open(str(file_path))
        text_parts: list[str] = []
        images: list[str] = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text_parts.append(page.get_text())

            # Extract embedded images
            for img_idx, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n > 4:  # CMYK
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    img_name = f"page{page_num + 1}_img{img_idx + 1}.png"
                    # Caller should set the actual save path
                    images.append(img_name)
                except Exception as e:
                    logger.debug(f"Failed to extract image xref={xref}: {e}")

        doc.close()
        full_text = "\n\n".join(text_parts)
        logger.info(f"PDF loaded: {file_path.name}, {len(full_text)} chars, {len(images)} images")
        return DocumentContent(text=full_text, images=images)
