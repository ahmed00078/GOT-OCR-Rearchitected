# File: services/ocr_service.py
"""
OCR Service for GOT-OCR 2.0
Handles OCR processing logic and file management
"""

import logging
import os
import shutil
import tempfile
import uuid
import base64
from typing import List, Optional, Dict, Any

from fastapi import UploadFile, BackgroundTasks
from PIL import Image

from models.ocr_model import OCRModelManager
from config import Config
from render import render_ocr_text

logger = logging.getLogger(__name__)


class OCRService:
    """Service class for OCR processing operations"""
    
    def __init__(self, model_manager: OCRModelManager, config: Config):
        self.model_manager = model_manager
        self.config = config
    
    async def process_images(
        self,
        task: str,
        images: List[UploadFile],
        background_tasks: BackgroundTasks,
        ocr_type: Optional[str] = None,
        ocr_box: Optional[str] = None,
        ocr_color: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process uploaded images with OCR"""
        
        # Create temporary workspace
        temp_dir = tempfile.mkdtemp()
        background_tasks.add_task(self._cleanup_tempdir, temp_dir)
        
        try:
            # Save uploaded images
            image_paths = await self._save_uploaded_images(images, temp_dir)
            
            # Load images
            pil_images = [Image.open(img_path) for img_path in image_paths]
            
            # Prepare task configuration
            task_config = self._prepare_task_config(task, ocr_type, ocr_box, ocr_color)
            
            # Generate OCR text
            text_result = self.model_manager.generate_text(pil_images, task_config)
            
            # Generate unique result ID
            result_id = str(uuid.uuid4())
            
            # Handle rendering for formatted outputs
            html_content = None
            if self._requires_html_rendering(task):
                html_content = await self._render_html_output(
                    text_result, temp_dir, result_id, task_config
                )
            
            # Prepare response
            response = {
                "result_id": result_id,
                "text": text_result,
                "html_available": html_content is not None
            }
            
            if html_content:
                encoded_html = base64.b64encode(html_content.encode()).decode()
                response.update({
                    "html": encoded_html,
                    "download_url": f"/results/{result_id}"
                })
            
            logger.info(f"Successfully processed {len(images)} images for task: {task}")
            return response
            
        except Exception as e:
            logger.error(f"OCR processing failed: {str(e)}")
            raise
    
    async def _save_uploaded_images(
        self, 
        images: List[UploadFile], 
        temp_dir: str
    ) -> List[str]:
        """Save uploaded images to temporary directory"""
        image_paths = []
        
        for img in images:
            # Validate file type
            if not img.content_type.startswith("image/"):
                raise ValueError(f"Invalid file type: {img.content_type}")
            
            # Save file
            img_path = os.path.join(temp_dir, img.filename)
            with open(img_path, "wb") as buffer:
                content = await img.read()
                buffer.write(content)
            
            image_paths.append(img_path)
            logger.debug(f"Saved image: {img.filename}")
        
        return image_paths
    
    def _prepare_task_config(
        self,
        task: str,
        ocr_type: Optional[str] = None,
        ocr_box: Optional[str] = None,
        ocr_color: Optional[str] = None
    ) -> Dict[str, Any]:
        """Prepare task configuration for model processing"""
        config = {"task": task}
        
        if ocr_type:
            config["ocr_type"] = ocr_type
        
        if ocr_box:
            # Parse box coordinates
            try:
                box = list(map(int, ocr_box.strip('[]').split(',')))
                if len(box) != 4:
                    raise ValueError("Box must contain exactly 4 coordinates")
                config["box"] = box
            except (ValueError, AttributeError) as e:
                raise ValueError(f"Invalid box format: {ocr_box}. Use [x1,y1,x2,y2]")
        
        if ocr_color:
            if ocr_color not in self.config.SUPPORTED_OCR_COLORS:
                raise ValueError(f"Invalid color: {ocr_color}. Supported: {self.config.SUPPORTED_OCR_COLORS}")
            config["color"] = ocr_color
        
        return config
    
    def _requires_html_rendering(self, task: str) -> bool:
        """Check if task requires HTML rendering"""
        return any(keyword in task for keyword in ["Format", "Fine-grained", "Multi"])
    
    async def _render_html_output(
        self,
        text_result: str,
        temp_dir: str,
        result_id: str,
        task_config: Dict[str, Any]
    ) -> Optional[str]:
        """Render OCR text to HTML format"""
        try:
            result_path = os.path.join(temp_dir, f"{result_id}.html")
            
            # Determine if formatting is required
            format_text = task_config.get("ocr_type") == "format"
            
            # Render text to HTML
            render_ocr_text(text_result, result_path, format_text=format_text)
            
            # Read rendered HTML
            if os.path.exists(result_path):
                with open(result_path, "r", encoding="utf-8") as f:
                    return f.read()
            
            return None
            
        except Exception as e:
            logger.warning(f"HTML rendering failed: {str(e)}")
            return None
    
    async def _cleanup_tempdir(self, temp_dir: str) -> None:
        """Clean up temporary directory"""
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.debug(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temporary directory {temp_dir}: {str(e)}")