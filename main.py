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
from services.reasoning_service import ExtractionType
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
    models_used: Optional[Dict[str, str]] = None

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
        
        # === ENDPOINT OCR STANDARD (INCHANGÉ) ===
        @app.post("/process", response_model=OCRResponse, 
                 summary="Standard OCR processing",
                 response_description="OCR processing results with text and optional HTML output")
        async def process_ocr(
            background_tasks: BackgroundTasks,
            task: str = Form(..., description=self.config.TASK_DESCRIPTIONS["task"]),
            ocr_type: Optional[str] = Form(None, description=self.config.TASK_DESCRIPTIONS["ocr_type"]),
            ocr_box: Optional[str] = Form(None, description=self.config.TASK_DESCRIPTIONS["ocr_box"]),
            ocr_color: Optional[str] = Form(None, description=self.config.TASK_DESCRIPTIONS["ocr_color"]),
            images: List[UploadFile] = File(..., description=self.config.TASK_DESCRIPTIONS["images"])
        ):
            """Standard OCR processing endpoint"""
            
            # Validate request
            validation_result = self.validator.validate_request(task, ocr_type, ocr_box, ocr_color, images)
            if not validation_result.is_valid:
                raise HTTPException(400, detail=validation_result.error_message)
            
            # Process request
            try:
                result = await self.enhanced_ocr_service.process_images(
                    task=task,
                    images=images,
                    ocr_type=ocr_type,
                    ocr_box=ocr_box,
                    ocr_color=ocr_color,
                    background_tasks=background_tasks
                )
                
                return OCRResponse(**result)
                
            except Exception as e:
                logger.error(f"OCR processing failed: {str(e)}")
                raise HTTPException(500, detail=f"Processing error: {str(e)}")
        
        # === NOUVEAU: ENDPOINT OCR + IA ===
        @app.post("/smart-extract", response_model=EnhancedOCRResponse,
                 summary="🧠 Smart OCR + AI Information Extraction",
                 response_description="OCR + AI reasoning results with structured data extraction")
        async def smart_extract(
            background_tasks: BackgroundTasks,
            task: str = Form("Multi-page OCR", description=self.config.TASK_DESCRIPTIONS["task"]),
            extraction_type: str = Form(..., description=self.config.TASK_DESCRIPTIONS["extraction_type"]),
            custom_instructions: Optional[str] = Form(None, description=self.config.TASK_DESCRIPTIONS["custom_instructions"]),
            ocr_type: Optional[str] = Form("format", description=self.config.TASK_DESCRIPTIONS["ocr_type"]),
            ocr_box: Optional[str] = Form(None, description=self.config.TASK_DESCRIPTIONS["ocr_box"]),
            ocr_color: Optional[str] = Form(None, description=self.config.TASK_DESCRIPTIONS["ocr_color"]),
            images: List[UploadFile] = File(..., description=self.config.TASK_DESCRIPTIONS["images"])
        ):
            """🧠 Smart extraction: OCR + AI reasoning for structured data extraction"""
            
            # Vérifications préliminaires
            if not self.enhanced_ocr_service:
                raise HTTPException(500, detail="Enhanced OCR service not available")
            
            if not self.config.reasoning_enabled:
                raise HTTPException(400, detail="AI reasoning is disabled")
            
            if not self.config.validate_extraction_type(extraction_type):
                raise HTTPException(400, detail=f"Invalid extraction type: {extraction_type}")
            
            # Validation standard
            validation_result = self.validator.validate_request(task, ocr_type, ocr_box, ocr_color, images)
            if not validation_result.is_valid:
                raise HTTPException(400, detail=validation_result.error_message)
            
            # Validation pour custom extraction
            if extraction_type == "custom" and not custom_instructions:
                raise HTTPException(400, detail="Custom instructions required for custom extraction type")
            
            # Traitement avec raisonnement
            try:
                logger.info(f"Smart extraction: {task} → {extraction_type}")
                
                result = await self.enhanced_ocr_service.process_with_reasoning(
                    task=task,
                    images=images,
                    background_tasks=background_tasks,
                    extraction_type=extraction_type,
                    custom_instructions=custom_instructions,
                    ocr_type=ocr_type,
                    ocr_box=ocr_box,
                    ocr_color=ocr_color
                )
                
                return EnhancedOCRResponse(**result)
                
            except Exception as e:
                logger.error(f"Smart extraction failed: {str(e)}")
                raise HTTPException(500, detail=f"Smart extraction error: {str(e)}")
        
        # === NOUVEAU: ENDPOINT BATCH PROCESSING ===
        @app.post("/batch-extract", 
                 summary="🚀 Batch Smart Extraction",
                 response_description="Batch processing for multiple documents")
        async def batch_smart_extract(
            # Note: Pour la simplicité, on utilise un endpoint simplifié
            # En production, on pourrait utiliser un format plus complexe
            extraction_type: str = Form(...),
            images: List[UploadFile] = File(...)
        ):
            """Batch processing for multiple documents (simplified version)"""
            
            if not self.enhanced_ocr_service:
                raise HTTPException(500, detail="Enhanced OCR service not available")
            
            if len(images) > 10:
                raise HTTPException(400, detail="Maximum 10 files for batch processing")
            
            try:
                # Créer des batches simples (un fichier par batch pour cette démo)
                files_batch = [[img] for img in images]
                configs = [{"extraction_type": extraction_type, "task": "Multi-page OCR"} for _ in images]
                
                results = await self.enhanced_ocr_service.batch_process_with_reasoning(
                    files_batch, configs
                )
                
                return {
                    "batch_results": results,
                    "total_processed": len(results),
                    "successful": sum(1 for r in results if "error" not in r)
                }
                
            except Exception as e:
                logger.error(f"Batch processing failed: {str(e)}")
                raise HTTPException(500, detail=f"Batch processing error: {str(e)}")
        
        # === NOUVEAU: ENDPOINT DE DÉMONSTRATION ===
        @app.get("/demo", summary="🎯 Demo endpoints and capabilities")
        async def demo_info():
            """Information about demo capabilities and example usage"""
            return {
                "demo_endpoints": {
                    "/process": "Standard OCR processing",
                    "/smart-extract": "🧠 OCR + AI reasoning",
                    "/batch-extract": "🚀 Batch processing",
                    "/extraction-types": "Available extraction types"
                },
                "example_workflows": {
                    "carbon_footprint": {
                        "description": "Extract environmental data from product sheets",
                        "endpoint": "/smart-extract",
                        "extraction_type": "carbon_footprint",
                        "example_output": {
                            "carbon_emissions": "45g CO2 eq",
                            "energy_consumption": "15W",
                            "product_name": "Laptop Model X"
                        }
                    },
                    "technical_specs": {
                        "description": "Extract technical specifications",
                        "endpoint": "/smart-extract", 
                        "extraction_type": "technical_specs",
                        "example_output": {
                            "product_name": "Smartphone Pro",
                            "dimensions": "150x75x8mm",
                            "weight": "180g"
                        }
                    }
                },
                "performance_tips": [
                    "Use 'Multi-page OCR' task for best results",
                    "Provide clear, high-resolution images",
                    "Custom instructions should be specific",
                    "Batch processing is optimized for up to 10 files"
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