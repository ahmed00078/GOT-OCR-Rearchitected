# File: main.py - Enhanced with SmolLM2 integration
"""
GOT-OCR 2.0 + SmolLM2 API Service
Advanced OCR + AI Reasoning microservice
"""

import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import Config
from models.ocr_model import OCRModelManager
from services.enhanced_ocr_service import EnhancedOCRService
from services.reasoning_service import SmolLM2ReasoningService
from utils.validators import OCRRequestValidator
from utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__)

# === NOUVEAUX MODÈLES DE RÉPONSE ===

class OCRResponse(BaseModel):
    """Response model for standard OCR processing"""
    result_id: str
    text: str
    html_available: bool
    html: Optional[str] = None
    download_url: Optional[str] = None
    multipage_info: Optional[Dict[str, Any]] = None

class EnhancedOCRResponse(BaseModel):
    """Response model for OCR + Reasoning processing"""
    result_id: str
    text: str
    html_available: bool
    html: Optional[str] = None
    download_url: Optional[str] = None
    multipage_info: Optional[Dict[str, Any]] = None
    
    # Nouvelles données de raisonnement
    reasoning_enabled: bool
    extraction_type: Optional[str] = None
    extraction_result: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    models_used: Optional[Dict[str, Optional[str]]] = None

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    device: str
    ocr_model_loaded: bool
    reasoning_model_loaded: bool
    gpu_available: bool
    version: str

class ExtractionTypesResponse(BaseModel):
    """Available extraction types response"""
    extraction_types: Dict[str, Dict[str, Any]]
    reasoning_available: bool

class GOTOCREnhancedApp:
    """Main application class for GOT-OCR 2.0 + SmolLM2 API"""
    
    def __init__(self):
        self.config = Config()
        self.app = self._create_app()
        self.model_manager: Optional[OCRModelManager] = None
        self.enhanced_ocr_service: Optional[EnhancedOCRService] = None
        self.validator = OCRRequestValidator()
        
    def _create_app(self) -> FastAPI:
        """Create and configure FastAPI application"""
        app = FastAPI(
            title=self.config.APP_TITLE,
            description=self.config.APP_DESCRIPTION,
            version=self.config.APP_VERSION,
            contact=self.config.CONTACT_INFO,
            license_info=self.config.LICENSE_INFO
        )
        
        # Add middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.CORS_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Mount static files
        app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")
        app.mount("/static", StaticFiles(directory="static"), name="static")
        
        # Register event handlers
        app.add_event_handler("startup", self._startup)
        app.add_event_handler("shutdown", self._shutdown)
        
        # Register routes
        self._register_routes(app)
        
        return app
    
    async def _startup(self):
        """Application startup handler"""
        logger.info("Starting GOT-OCR 2.0 + SmolLM2 API service...")
        
        try:
            # Initialize OCR model manager
            self.model_manager = OCRModelManager(self.config)
            await self.model_manager.load_model()
            
            # Initialize enhanced OCR service
            self.enhanced_ocr_service = EnhancedOCRService(self.model_manager, self.config)
            
            # Initialize reasoning model (asynchronously)
            if self.config.reasoning_enabled:
                logger.info("Initialisation du modèle de raisonnement...")
                await self.enhanced_ocr_service.initialize_reasoning()
            else:
                logger.info("Raisonnement IA désactivé")
            
            logger.info(f"Service démarré avec succès sur device: {self.model_manager.device}")
            
        except Exception as e:
            logger.error(f"Failed to start service: {str(e)}")
            raise
    
    async def _shutdown(self):
        """Application shutdown handler"""
        logger.info("Shutting down GOT-OCR 2.0 + SmolLM2 API service...")
        
        if self.enhanced_ocr_service:
            await self.enhanced_ocr_service.cleanup()
            
        if self.model_manager:
            await self.model_manager.cleanup()
    
    def _register_routes(self, app: FastAPI):
        """Register API routes"""
        
        @app.get("/", summary="Root endpoint")
        async def root():
            return {
                "message": "GOT-OCR 2.0 + SmolLM2 API is running",
                "docs": "/docs",
                "frontend": "/frontend/index.html",
                "health": "/health",
                "api_version": self.config.APP_VERSION,
                "features": [
                    "Multi-page OCR",
                    "PDF Support", 
                    "AI-powered information extraction",
                    "SmolLM2:1.7B reasoning"
                ]
            }
        
        @app.get("/health", response_model=HealthResponse, summary="Health check")
        async def health_check():
            """Health check endpoint with detailed system information"""
            reasoning_loaded = (
                self.enhanced_ocr_service is not None and 
                self.enhanced_ocr_service.is_reasoning_available
            )
            
            return HealthResponse(
                status="healthy",
                device=str(self.model_manager.device) if self.model_manager else "unknown",
                ocr_model_loaded=self.model_manager is not None and self.model_manager.is_loaded,
                reasoning_model_loaded=reasoning_loaded,
                gpu_available=torch.cuda.is_available(),
                version=self.config.APP_VERSION
            )
        
        # === ENDPOINT UNIFIÉ POUR TOUT TRAITEMENT ===
        @app.post("/process", response_model=EnhancedOCRResponse, 
                 summary="Unified OCR Processing",
                 response_description="Unified endpoint for OCR processing with optional AI reasoning")
        async def process_unified(
            background_tasks: BackgroundTasks,
            task: str = Form(..., description=self.config.TASK_DESCRIPTIONS["task"]),
            ocr_type: Optional[str] = Form(None, description=self.config.TASK_DESCRIPTIONS["ocr_type"]),
            ocr_box: Optional[str] = Form(None, description=self.config.TASK_DESCRIPTIONS["ocr_box"]),
            ocr_color: Optional[str] = Form(None, description=self.config.TASK_DESCRIPTIONS["ocr_color"]),
            enable_reasoning: bool = Form(False, description="Enable AI reasoning for data extraction"),
            custom_instructions: Optional[str] = Form(None, description="Custom AI extraction instructions (only when reasoning enabled)"),
            images: List[UploadFile] = File(..., description=self.config.TASK_DESCRIPTIONS["images"])
        ):
            """Unified OCR processing endpoint - handles both simple OCR and AI-enhanced processing"""
            
            # Validate basic request
            validation_result = self.validator.validate_request(task, ocr_type, ocr_box, ocr_color, images)
            if not validation_result.is_valid:
                raise HTTPException(400, detail=validation_result.error_message)
            
            # Validate AI reasoning parameters
            if enable_reasoning:
                if not self.enhanced_ocr_service:
                    raise HTTPException(500, detail="Enhanced OCR service not available")
                
                if not self.config.reasoning_enabled:
                    raise HTTPException(400, detail="AI reasoning is disabled in configuration")
                
                if not custom_instructions or len(custom_instructions.strip()) < 10:
                    raise HTTPException(400, detail="Custom instructions required (minimum 10 characters) when reasoning is enabled")
            
            # Process request
            try:
                if enable_reasoning:
                    # OCR + AI processing
                    logger.info(f"Processing with AI reasoning: {task}")
                    result = await self.enhanced_ocr_service.process_with_reasoning(
                        task=task,
                        images=images,
                        background_tasks=background_tasks,
                        extraction_type="custom",  # Always use custom
                        custom_instructions=custom_instructions,
                        ocr_type=ocr_type,
                        ocr_box=ocr_box,
                        ocr_color=ocr_color
                    )
                else:
                    # Standard OCR only
                    logger.info(f"Processing standard OCR: {task}")
                    result = await self.enhanced_ocr_service.process_images(
                        task=task,
                        images=images,
                        ocr_type=ocr_type,
                        ocr_box=ocr_box,
                        ocr_color=ocr_color,
                        background_tasks=background_tasks
                    )
                    
                    # Add reasoning fields as None for consistent response
                    result.update({
                        "reasoning_enabled": False,
                        "extraction_type": None,
                        "extraction_result": None,
                        "performance_metrics": {
                            "total_time": result.get("processing_time", 0),
                            "text_length": len(result.get("text", ""))
                        },
                        "models_used": {
                            "ocr_model": self.model_manager.config.MODEL_NAME,
                            "reasoning_model": None
                        }
                    })
                
                return EnhancedOCRResponse(**result)
                
            except Exception as e:
                logger.error(f"Unified processing failed: {str(e)}")
                raise HTTPException(500, detail=f"Processing error: {str(e)}")
        
        # === ENDPOINT DE DÉMONSTRATION ===
        @app.get("/demo", summary="🎯 Demo capabilities")
        async def demo_info():
            """Information about demo capabilities and example usage"""
            return {
                "unified_endpoint": "/process",
                "usage_modes": {
                    "ocr_only": {
                        "description": "Standard OCR processing",
                        "parameters": {
                            "enable_reasoning": False,
                            "task": "Plain text OCR / Format text OCR / Multi-page OCR",
                            "ocr_type": "plain / format"
                        }
                    },
                    "ocr_with_ai": {
                        "description": "OCR + AI custom extraction",
                        "parameters": {
                            "enable_reasoning": True,
                            "custom_instructions": "Extract product specifications, prices, contact information, etc."
                        },
                        "example_instructions": [
                            "Extract all product names, prices, and technical specifications",
                            "Find contact information including emails, phones, and addresses",
                            "Extract carbon footprint data and environmental certifications",
                            "Get financial data including costs, revenues, and currencies"
                        ]
                    }
                },
                "performance_tips": [
                    "Use 'Multi-page OCR' task for best results",
                    "Provide clear, high-resolution images",
                    "Custom instructions should be specific (minimum 10 characters)",
                    "Set enable_reasoning=true only when you need AI extraction"
                ]
            }

# Create application instance
app_instance = GOTOCREnhancedApp()
app = app_instance.app

if __name__ == "__main__":
    import uvicorn
    
    # Configure uvicorn logging
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        log_config=log_config,
        access_log=True
    )