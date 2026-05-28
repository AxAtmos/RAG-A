from __future__ import annotations

from pathlib import Path

from loguru import logger

from .loader_registry import LoaderRegistry, DocumentContent


@LoaderRegistry.register([".png", ".jpg", ".jpeg", ".bmp", ".tiff"])
class OcrLoader:
    """OCR loader using PaddleOCR (optional dependency)."""

    def __init__(self):
        self._ocr = None

    def _get_ocr(self):
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR
                self._ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
                logger.info("PaddleOCR initialized")
            except ImportError:
                logger.warning("PaddleOCR not installed, OCR unavailable")
                raise
        return self._ocr

    def load(self, file_path: Path) -> DocumentContent:
        try:
            ocr = self._get_ocr()
            result = ocr.ocr(str(file_path), cls=True)
            texts: list[str] = []
            if result and result[0]:
                for line in result[0]:
                    text = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                    texts.append(text)
            full_text = "\n".join(texts)
            logger.info(f"OCR completed: {file_path.name}, {len(full_text)} chars")
            return DocumentContent(text=full_text, images=[file_path.name])
        except ImportError:
            logger.error("PaddleOCR not available, returning empty content")
            return DocumentContent(text=f"[OCR unavailable: {file_path.name}]")
        except Exception as e:
            logger.error(f"OCR failed for {file_path.name}: {e}")
            return DocumentContent(text="")
