# File: services/__init__.py
"""
Services package for GOT-OCR 2.0 API
"""

from .ocr_service import OCRService
from .pdf_service import PDFService

__all__ = ["OCRService", "PDFService"]