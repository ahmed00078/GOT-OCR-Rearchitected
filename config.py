import os
from typing import List, Dict, Any
from globe import description, tasks, ocr_types, ocr_colors
import torch


class Config:
    """Configuration class for GOT-OCR 2.0 + AI Model API"""
    
    APP_TITLE: str = "GOT-OCR 2.0 + AI Model API"
    APP_DESCRIPTION: str = description + "\n\n🧠 **Enhanced with AI model** for intelligent information extraction!"
    APP_VERSION: str = "2.2"
    
    CONTACT_INFO: Dict[str, str] = {
        "name": "Ahmed Sidi Mohammed",
        "email": "ahmedsidimohammed78@gmail.com"
    }
    
    LICENSE_INFO: Dict[str, str] = {
        "name": "Apache 2.0",
        "url": "https://github.com/ahmed00078/GOT-OCR-Rearchitected/blob/main/LICENSE.txt"
    }
    
    # CORS settings
    CORS_ORIGINS: List[str] = ["*"]
    
    # === OCR MODEL SETTINGS ===
    MODEL_NAME: str = "stepfun-ai/GOT-OCR-2.0-hf"
    MAX_NEW_TOKENS: int = 4096
    STOP_STRINGS: str = "<|im_end|>"
    
    # === NOUVEAU: AI Model SETTINGS ===
    REASONING_MODEL_NAME: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"  # qwen3:8b for Ollama model name
    REASONING_MAX_TOKENS: int = 4096
    REASONING_TEMPERATURE: float = 0.1
    REASONING_BATCH_SIZE: int = 3 
    
    # Ollama settings
    OLLAMA_URL: str = "http://localhost:11434"
    
    # Enable/disable reasoning features
    ENABLE_REASONING: bool = True
    ENABLE_QUANTIZATION: bool = True    # Not used with Ollama
    
    # Processing settings
    SUPPORTED_TASKS: List[str] = tasks
    SUPPORTED_OCR_TYPES: List[str] = ocr_types
    SUPPORTED_OCR_COLORS: List[str] = ocr_colors
    
    # === TYPES D'EXTRACTION SIMPLIFIÉS ===
    SUPPORTED_EXTRACTION_TYPES: List[str] = [
        "custom"
    ]
    
    # PDF Processing settings (inchangé)
    MAX_FILE_SIZE: int = 50 * 1024 * 1024
    MAX_PDF_SIZE: int = 100 * 1024 * 1024 
    SUPPORTED_FORMATS: List[str] = [
        "image/jpeg", 
        "image/png", 
        "image/tiff",
        "application/pdf"
    ]
    
    PDF_CONVERSION_DPI: int = 300
    PDF_MAX_PAGES: int = 50
    PDF_OUTPUT_FORMAT: str = "PNG"
    PDF_MEMORY_LIMIT_MB: int = 500
    
    # Multi-page specific settings
    MULTIPAGE_MAX_FILES: int = 20
    MULTIPAGE_BATCH_SIZE: int = 5
    MULTIPAGE_CONCAT_SEPARATOR: str = "\n\n--- Page {} ---\n\n"
    
    # Performance settings
    LOW_CPU_MEM_USAGE: bool = True
    USE_GPU_IF_AVAILABLE: bool = True
    
    # Timeout settings
    UVICORN_TIMEOUT: int = int(os.getenv("UVICORN_TIMEOUT", "600"))
    PDF_CONVERSION_TIMEOUT: int = 300
    
    # Environment optimization
    OMP_NUM_THREADS: int = int(os.getenv("OMP_NUM_THREADS", "4"))
    MKL_NUM_THREADS: int = int(os.getenv("MKL_NUM_THREADS", "4"))
    NUMEXPR_NUM_THREADS: int = int(os.getenv("NUMEXPR_NUM_THREADS", "4"))
    TOKENIZERS_PARALLELISM: bool = os.getenv("TOKENIZERS_PARALLELISM", "true").lower() == "true"
    
    # === NOUVELLES DESCRIPTIONS POUR L'API ===
    TASK_DESCRIPTIONS: Dict[str, str] = {
        "task": (
            "Select the type of OCR processing to perform. Available options:\n\n"
            "- **Plain Text OCR**: Basic text extraction from images\n"
            "- **Format Text OCR**: Structured text output (LaTeX/Markdown)\n"
            "- **Fine-grained OCR (Box)**: Extract text from specific regions using coordinates\n"
            "- **Fine-grained OCR (Color)**: Extract text from color-highlighted regions\n"
            "- **Multi-crop OCR**: Process multiple image regions automatically\n"
            "- **Multi-page OCR**: Process multi-page documents (PDFs or image sequences)\n"
            "- **Smart Extract**: OCR + AI reasoning for structured data extraction"
        ),
        "custom_instructions": (
            "Provide specific AI extraction instructions:\n\n"
            "Examples:\n"
            "- 'Extract all product names, prices, and warranty information'\n"
            "- 'Extract carbon footprint data and environmental certifications'\n"
            "Be specific about what data you want and the desired output format."
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
            "**🧠 Smart Processing:**\n"
            "- Automatic OCR + AI reasoning\n"
            "- Structured data extraction\n"
            "- High confidence scoring"
        )
    }
    
    # === PARAMÈTRES SPÉCIFIQUES RAISONNEMENT ===
    REASONING_CONFIG: Dict[str, Any] = {
        "max_context_length": 4096
    }
    
    @property
    def device_preference(self) -> str:
        """Get device preference based on configuration and availability"""
        if self.USE_GPU_IF_AVAILABLE:
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cpu"
    
    @property
    def reasoning_enabled(self) -> bool:
        """Check if reasoning features are enabled"""
        return self.ENABLE_REASONING
    
    # === NOUVELLES MÉTHODES UTILITAIRES ===
    
    def get_reasoning_settings(self) -> Dict[str, Any]:
        """Get all reasoning-related settings"""
        return {
            "model_name": self.REASONING_MODEL_NAME,
            "max_tokens": self.REASONING_MAX_TOKENS,
            "temperature": self.REASONING_TEMPERATURE,
            "batch_size": self.REASONING_BATCH_SIZE,
            "enable_quantization": self.ENABLE_QUANTIZATION,
            "config": self.REASONING_CONFIG
        }
    
    def validate_extraction_type(self, extraction_type: str) -> bool:
        """Validate extraction type"""
        return extraction_type in self.SUPPORTED_EXTRACTION_TYPES