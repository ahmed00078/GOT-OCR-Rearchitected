# File: config.py - Version Multi-Page Enhanced
"""
Configuration settings for GOT-OCR 2.0 API - Multi-Page Support
"""

import os
from typing import List, Dict, Any
from globe import title, description, tasks, ocr_types, ocr_colors
import torch


class Config:
    """Configuration class for GOT-OCR 2.0 API - Enhanced for Multi-Page"""
    
    # Application settings
    APP_TITLE: str = "GOT-OCR 2.0 API - Multi-Page Edition"
    APP_DESCRIPTION: str = description
    APP_VERSION: str = "2.1"  # Version incrémentée pour multi-page
    
    # Contact and license information
    CONTACT_INFO: Dict[str, str] = {
        "name": "API Support",
        "email": "ahmedsidimohammed78@gmail.com"
    }
    
    LICENSE_INFO: Dict[str, str] = {
        "name": "Apache 2.0",
        "url": "https://github.com/ahmed00078/GOT-OCR2.0/blob/main/LICENSE.txt"
    }
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["*"]
    
    # Model settings
    MODEL_NAME: str = "stepfun-ai/GOT-OCR-2.0-hf"
    MAX_NEW_TOKENS: int = 4096
    STOP_STRINGS: str = "<|im_end|>"
    
    # Processing settings
    SUPPORTED_TASKS: List[str] = tasks
    SUPPORTED_OCR_TYPES: List[str] = ocr_types
    SUPPORTED_OCR_COLORS: List[str] = ocr_colors
    
    # === NOUVEAUX PARAMÈTRES POUR MULTI-PAGE ===
    
    # File upload settings - Enhanced for PDF
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB max pour images
    MAX_PDF_SIZE: int = 100 * 1024 * 1024  # 100MB max pour PDFs
    SUPPORTED_FORMATS: List[str] = [
        "image/jpeg", 
        "image/png", 
        "image/tiff",
        "application/pdf"  # === NOUVEAU FORMAT SUPPORTÉ ===
    ]
    
    # PDF Processing settings
    PDF_CONVERSION_DPI: int = 300  # DPI pour conversion PDF->Image
    PDF_MAX_PAGES: int = 50        # Limite de pages par PDF
    PDF_OUTPUT_FORMAT: str = "PNG" # Format d'image pour pages converties
    PDF_MEMORY_LIMIT_MB: int = 500 # Limite mémoire pour gros PDFs
    
    # Multi-page specific settings
    MULTIPAGE_MAX_FILES: int = 20          # Max 20 fichiers par requête
    MULTIPAGE_BATCH_SIZE: int = 5          # Traitement par batch de 5 pages
    MULTIPAGE_CONCAT_SEPARATOR: str = "\n\n--- Page {} ---\n\n"  # Séparateur entre pages
    
    # Performance settings
    LOW_CPU_MEM_USAGE: bool = True
    USE_GPU_IF_AVAILABLE: bool = True
    
    # Timeout settings - Augmentés pour multi-page
    UVICORN_TIMEOUT: int = int(os.getenv("UVICORN_TIMEOUT", "600"))  # 10 minutes
    PDF_CONVERSION_TIMEOUT: int = 300  # 5 minutes max pour conversion PDF
    
    # Environment optimization
    OMP_NUM_THREADS: int = int(os.getenv("OMP_NUM_THREADS", "4"))
    MKL_NUM_THREADS: int = int(os.getenv("MKL_NUM_THREADS", "4"))
    NUMEXPR_NUM_THREADS: int = int(os.getenv("NUMEXPR_NUM_THREADS", "4"))
    TOKENIZERS_PARALLELISM: bool = os.getenv("TOKENIZERS_PARALLELISM", "true").lower() == "true"
    
    # === DESCRIPTIONS MISES À JOUR POUR L'API ===
    TASK_DESCRIPTIONS: Dict[str, str] = {
        "task": (
            "Select the type of OCR processing to perform. Available options:\n\n"
            "- **Plain Text OCR**: Basic text extraction from images\n"
            "- **Format Text OCR**: Structured text output (LaTeX/Markdown)\n"
            "- **Fine-grained OCR (Box)**: Extract text from specific regions using coordinates\n"
            "- **Fine-grained OCR (Color)**: Extract text from color-highlighted regions\n"
            "- **Multi-crop OCR**: Process multiple image regions automatically\n"
            "- **Multi-page OCR**: 🆕 Process multi-page documents (PDFs or image sequences)"
        ),
        "ocr_type": (
            "Required for formatted outputs. Use 'format' to enable structured text output.\n\n"
            "Applies to:\n"
            "- Format Text OCR\n"
            "- Multi-crop OCR\n"
            "- Multi-page OCR"
        ),
        "ocr_box": (
            "Required for box-based extraction. Format as [x1,y1,x2,y2] where:\n\n"
            "- x1: Top-left X coordinate\n"
            "- y1: Top-left Y coordinate\n"
            "- x2: Bottom-right X coordinate\n"
            "- y2: Bottom-right Y coordinate\n\n"
            "Example: [100,200,300,400]"
        ),
        "ocr_color": "Select color for region-based extraction (red, green, blue)",
        "images": (
            "Upload files for processing. Supported formats:\n\n"
            "**Images:**\n"
            "- JPEG/JPG (max 50MB)\n"
            "- PNG (max 50MB)\n"
            "- TIFF (max 50MB)\n\n"
            "**Documents:** 🆕\n"
            "- PDF (max 100MB, up to 50 pages)\n\n"
            "**Multi-page tips:**\n"
            "- Upload ONE PDF for automatic page extraction\n"
            "- Or upload multiple images in correct order\n"
            "- Mix of PDFs and images supported but not recommended"
        )
    }
    
    # === NOUVEAUX PARAMÈTRES DE LOGGING ===
    ENABLE_PDF_LOGGING: bool = True
    LOG_PDF_CONVERSION_DETAILS: bool = True
    
    # === PARAMÈTRES DE CACHE (OPTIONNEL) ===
    ENABLE_PDF_CACHE: bool = False
    PDF_CACHE_DIR: str = "/tmp/got_ocr_pdf_cache"
    PDF_CACHE_EXPIRY_HOURS: int = 24
    
    @property
    def device_preference(self) -> str:
        """Get device preference based on configuration and availability"""
        if self.USE_GPU_IF_AVAILABLE:
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cpu"
    
    # === NOUVELLES MÉTHODES UTILITAIRES ===
    
    def get_pdf_settings(self) -> Dict[str, Any]:
        """Get all PDF-related settings in one dictionary"""
        return {
            "max_size_mb": self.MAX_PDF_SIZE / (1024 * 1024),
            "max_pages": self.PDF_MAX_PAGES,
            "conversion_dpi": self.PDF_CONVERSION_DPI,
            "output_format": self.PDF_OUTPUT_FORMAT,
            "timeout_seconds": self.PDF_CONVERSION_TIMEOUT,
            "memory_limit_mb": self.PDF_MEMORY_LIMIT_MB
        }
    
    def get_multipage_settings(self) -> Dict[str, Any]:
        """Get all multi-page related settings"""
        return {
            "max_files": self.MULTIPAGE_MAX_FILES,
            "batch_size": self.MULTIPAGE_BATCH_SIZE,
            "page_separator": self.MULTIPAGE_CONCAT_SEPARATOR,
            "supported_formats": self.SUPPORTED_FORMATS
        }
    
    def is_pdf_supported(self) -> bool:
        """Check if PDF support is enabled"""
        return "application/pdf" in self.SUPPORTED_FORMATS
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return any warnings"""
        warnings = []
        
        if self.PDF_MAX_PAGES > 100:
            warnings.append("PDF_MAX_PAGES > 100 may cause memory issues")
        
        if self.PDF_CONVERSION_DPI > 400:
            warnings.append("PDF_CONVERSION_DPI > 400 may be slow")
        
        if self.MAX_PDF_SIZE > 200 * 1024 * 1024:  # 200MB
            warnings.append("MAX_PDF_SIZE > 200MB may cause timeouts")
        
        return warnings