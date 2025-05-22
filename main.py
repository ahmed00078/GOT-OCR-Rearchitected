# File: main.py
"""
GOT-OCR 2.0 API Service
Advanced OCR microservice powered by Transformers and FastAPI
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
from services.ocr_service import OCRService
from utils.validators import OCRRequestValidator
from utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__)

class OCRResponse(BaseModel):
    """Response model for OCR processing"""
    result_id: str
    text: str
    html_available: bool
    html: Optional[str] = None
    download_url: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    device: str
    model_loaded: bool
    gpu_available: bool

class GOTOCRApp:
    """Main application class for GOT-OCR 2.0 API"""
    
    def __init__(self):
        self.config = Config()
        self.app = self._create_app()
        self.model_manager: Optional[OCRModelManager] = None
        self.ocr_service: Optional[OCRService] = None
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
        logger.info("Starting GOT-OCR 2.0 API service...")
        
        try:
            # Initialize model manager
            self.model_manager = OCRModelManager(self.config)
            await self.model_manager.load_model()
            
            # Initialize OCR service
            self.ocr_service = OCRService(self.model_manager, self.config)
            
            logger.info(f"Service started successfully on device: {self.model_manager.device}")
            
        except Exception as e:
            logger.error(f"Failed to start service: {str(e)}")
            raise
    
    async def _shutdown(self):
        """Application shutdown handler"""
        logger.info("Shutting down GOT-OCR 2.0 API service...")
        
        if self.model_manager:
            await self.model_manager.cleanup()
    
    def _register_routes(self, app: FastAPI):
        """Register API routes"""
        
        @app.get("/", summary="Root endpoint")
        async def root():
            return {
                "message": "GOT-OCR 2.0 API is running",
                "docs": "/docs",
                "frontend": "/frontend/index.html",
                "Health": "/health",
                "api_version": self.config.APP_VERSION
            }
        
        @app.get("/health", response_model=HealthResponse, summary="Health check")
        async def health_check():
            """Health check endpoint with detailed system information"""
            return HealthResponse(
                status="healthy",
                device=str(self.model_manager.device) if self.model_manager else "unknown",
                model_loaded=self.model_manager is not None and self.model_manager.is_loaded,
                gpu_available=torch.cuda.is_available()
            )
        
        @app.post("/process", response_model=OCRResponse, 
                 summary="Process images for text extraction",
                 response_description="OCR processing results with text and optional HTML output")
        async def process_ocr(
            background_tasks: BackgroundTasks,
            task: str = Form(..., description=self.config.TASK_DESCRIPTIONS["task"]),
            ocr_type: Optional[str] = Form(None, description=self.config.TASK_DESCRIPTIONS["ocr_type"]),
            ocr_box: Optional[str] = Form(None, description=self.config.TASK_DESCRIPTIONS["ocr_box"]),
            ocr_color: Optional[str] = Form(None, description=self.config.TASK_DESCRIPTIONS["ocr_color"]),
            images: List[UploadFile] = File(..., description=self.config.TASK_DESCRIPTIONS["images"])
        ):
            """Main OCR processing endpoint supporting all GOT-OCR 2.0 features"""
            
            # Validate request
            validation_result = self.validator.validate_request(task, ocr_type, ocr_box, ocr_color, images)
            if not validation_result.is_valid:
                raise HTTPException(400, detail=validation_result.error_message)
            
            # Process request
            try:
                result = await self.ocr_service.process_images(
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
        
        @app.get("/results/{result_id}", 
                summary="Retrieve formatted OCR results",
                response_description="HTML-rendered OCR output")
        async def get_result(result_id: str):
            """Retrieve HTML-rendered results by ID"""
            # TODO: Implement result storage/retrieval logic
            return JSONResponse(
                content={"detail": "Result storage not implemented"}, 
                status_code=501
            )

# Create application instance
app_instance = GOTOCRApp()
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