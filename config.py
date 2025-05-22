# File: config.py
"""
Configuration settings for GOT-OCR 2.0 API
"""

import os
from typing import List, Dict, Any
from globe import title, description, tasks, ocr_types, ocr_colors
import torch


class Config:
    """Configuration class for GOT-OCR 2.0 API"""
    
    # Application settings
    APP_TITLE: str = "GOT-OCR 2.0 API"
    APP_DESCRIPTION: str = description
    APP_VERSION: str = "2.0"
    
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
    
    # File upload settings
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    SUPPORTED_FORMATS: List[str] = ["image/jpeg", "image/png", "image/tiff"]
    
    # Performance settings
    LOW_CPU_MEM_USAGE: bool = True
    USE_GPU_IF_AVAILABLE: bool = True
    
    # Timeout settings
    UVICORN_TIMEOUT: int = int(os.getenv("UVICORN_TIMEOUT", "300"))
    
    # Environment optimization
    OMP_NUM_THREADS: int = int(os.getenv("OMP_NUM_THREADS", "4"))
    MKL_NUM_THREADS: int = int(os.getenv("MKL_NUM_THREADS", "4"))
    NUMEXPR_NUM_THREADS: int = int(os.getenv("NUMEXPR_NUM_THREADS", "4"))
    TOKENIZERS_PARALLELISM: bool = os.getenv("TOKENIZERS_PARALLELISM", "true").lower() == "true"
    
    # Task descriptions for API documentation
    TASK_DESCRIPTIONS: Dict[str, str] = {
        "task": (
            "Select the type of OCR processing to perform. Available options:\n\n"
            "- **Plain Text OCR**: Basic text extraction from images\n"
            "- **Format Text OCR**: Structured text output (LaTeX/Markdown)\n"
            "- **Fine-grained OCR (Box)**: Extract text from specific regions using coordinates\n"
            "- **Fine-grained OCR (Color)**: Extract text from color-highlighted regions\n"
            "- **Multi-crop OCR**: Process multiple image regions automatically\n"
            "- **Multi-page OCR**: Process multi-page documents"
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
            "Upload image files for processing. Supported formats:\n\n"
            "- JPEG/JPG\n"
            "- PNG\n"
            "- TIFF\n\n"
            "For multi-page processing, upload multiple files in order"
        )
    }
    
    @property
    def device_preference(self) -> str:
        """Get device preference based on configuration and availability"""
        if self.USE_GPU_IF_AVAILABLE:
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cpu"