# File: config.py - Updated with SmolLM2 integration
"""
Configuration settings for GOT-OCR 2.0 API + SmolLM2 Reasoning
"""

import os
from typing import List, Dict, Any
from globe import title, description, tasks, ocr_types, ocr_colors
import torch


class Config:
    """Configuration class for GOT-OCR 2.0 + SmolLM2 API"""
    
    # Application settings
    APP_TITLE: str = "GOT-OCR 2.0 + SmolLM2 API"
    APP_DESCRIPTION: str = description + "\n\n🧠 **Enhanced with SmolLM2:1.7B** for intelligent information extraction!"
    APP_VERSION: str = "2.2"
    
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
    
    # === OCR MODEL SETTINGS ===
    MODEL_NAME: str = "stepfun-ai/GOT-OCR-2.0-hf"
    MAX_NEW_TOKENS: int = 4096
    STOP_STRINGS: str = "<|im_end|>"
    
    # === NOUVEAU: SMOLLM2 SETTINGS ===
    REASONING_MODEL_NAME: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    REASONING_MAX_TOKENS: int = 512
    REASONING_TEMPERATURE: float = 0.1  # Très déterministe
    REASONING_BATCH_SIZE: int = 3       # Petit batch pour efficacité
    
    # Enable/disable reasoning features
    ENABLE_REASONING: bool = True
    ENABLE_QUANTIZATION: bool = True    # 8-bit quantization pour SmolLM2
    REASONING_TIMEOUT: int = 30         # Timeout en secondes
    
    # Processing settings
    SUPPORTED_TASKS: List[str] = tasks
    SUPPORTED_OCR_TYPES: List[str] = ocr_types
    SUPPORTED_OCR_COLORS: List[str] = ocr_colors
    
    # === NOUVEAUX TYPES D'EXTRACTION ===
    SUPPORTED_EXTRACTION_TYPES: List[str] = [
        "carbon_footprint",
        "technical_specs", 
        "financial_data",
        "contact_info",
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
            "- **Multi-page OCR**: 🆕 Process multi-page documents (PDFs or image sequences)\n"
            "- **🧠 Smart Extract**: OCR + AI reasoning for structured data extraction"
        ),
        "extraction_type": (
            "🧠 **AI-Powered Information Extraction** (requires SmolLM2):\n\n"
            "- **carbon_footprint**: Extract environmental data (CO2, energy consumption)\n"
            "- **technical_specs**: Extract product specifications and technical details\n"
            "- **financial_data**: Extract prices, costs, financial metrics\n"
            "- **contact_info**: Extract names, emails, phones, addresses\n"
            "- **custom**: Custom extraction with your own instructions\n\n"
            "💡 Combines OCR with intelligent reasoning for structured output!"
        ),
        "custom_instructions": (
            "For 'custom' extraction type, provide specific instructions:\n\n"
            "Example: 'Extract all product names, prices, and warranty information'\n"
            "Be specific about what data you want and in what format."
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
        "enable_caching": True,           # Cache des résultats pour performance
        "max_context_length": 2000,      # Limite du contexte pour SmolLM2
        "confidence_threshold": 0.5,     # Seuil de confiance minimum
        "fallback_to_regex": True,       # Fallback regex si AI échoue
        "parallel_processing": False,    # Désactivé pour SmolLM2:1.7B
        "memory_optimization": True      # Optimisations mémoire
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
            "timeout": self.REASONING_TIMEOUT,
            "config": self.REASONING_CONFIG
        }
    
    def get_memory_settings(self) -> Dict[str, Any]:
        """Get memory optimization settings"""
        return {
            "low_cpu_mem_usage": self.LOW_CPU_MEM_USAGE,
            "enable_quantization": self.ENABLE_QUANTIZATION,
            "use_gpu": self.USE_GPU_IF_AVAILABLE,
            "pdf_memory_limit_mb": self.PDF_MEMORY_LIMIT_MB,
            "reasoning_max_context": self.REASONING_CONFIG["max_context_length"]
        }
    
    def validate_extraction_type(self, extraction_type: str) -> bool:
        """Validate extraction type"""
        return extraction_type in self.SUPPORTED_EXTRACTION_TYPES
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration for both OCR and reasoning"""
        return {
            "ocr_config": {
                "max_new_tokens": self.MAX_NEW_TOKENS,
                "low_cpu_mem_usage": self.LOW_CPU_MEM_USAGE,
                "multipage_batch_size": self.MULTIPAGE_BATCH_SIZE
            },
            "reasoning_config": {
                "max_tokens": self.REASONING_MAX_TOKENS,
                "temperature": self.REASONING_TEMPERATURE,
                "batch_size": self.REASONING_BATCH_SIZE,
                "enable_quantization": self.ENABLE_QUANTIZATION
            },
            "system_config": {
                "omp_threads": self.OMP_NUM_THREADS,
                "mkl_threads": self.MKL_NUM_THREADS,
                "device": self.device_preference
            }
        }
    
    def get_enhanced_task_list(self) -> List[str]:
        """Get task list including new AI-powered tasks"""
        enhanced_tasks = self.SUPPORTED_TASKS.copy()
        if self.reasoning_enabled:
            enhanced_tasks.append("Smart Extract (OCR + AI)")
        return enhanced_tasks