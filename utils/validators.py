# File: utils/validators.py
"""
Request validators for GOT-OCR 2.0 API
"""

import logging
from typing import List, Optional, NamedTuple
from fastapi import UploadFile

from config import Config

logger = logging.getLogger(__name__)


class ValidationResult(NamedTuple):
    """Result of request validation"""
    is_valid: bool
    error_message: Optional[str] = None


class OCRRequestValidator:
    """Validator for OCR processing requests"""
    
    def __init__(self):
        self.config = Config()
    
    def validate_request(
        self,
        task: str,
        ocr_type: Optional[str],
        ocr_box: Optional[str],
        ocr_color: Optional[str],
        images: List[UploadFile]
    ) -> ValidationResult:
        """Validate complete OCR request"""
        
        # Validate task
        task_validation = self._validate_task(task)
        if not task_validation.is_valid:
            return task_validation
        
        # Validate task-specific parameters
        params_validation = self._validate_task_parameters(task, ocr_type, ocr_box, ocr_color)
        if not params_validation.is_valid:
            return params_validation
        
        # Validate images
        images_validation = self._validate_images(images)
        if not images_validation.is_valid:
            return images_validation
        
        return ValidationResult(is_valid=True)
    
    def _validate_task(self, task: str) -> ValidationResult:
        """Validate task parameter"""
        if not task:
            return ValidationResult(False, "Task is required")
        
        if task not in self.config.SUPPORTED_TASKS:
            return ValidationResult(
                False, 
                f"Invalid task: {task}. Supported tasks: {', '.join(self.config.SUPPORTED_TASKS)}"
            )
        
        return ValidationResult(True)
    
    def _validate_task_parameters(
        self,
        task: str,
        ocr_type: Optional[str],
        ocr_box: Optional[str],
        ocr_color: Optional[str]
    ) -> ValidationResult:
        """Validate task-specific parameters"""
        
        # Validate Fine-grained OCR (Color)
        if task == "Fine-grained OCR (Color)":
            if not ocr_color:
                return ValidationResult(False, "Color parameter is required for Fine-grained OCR (Color)")
            
            if ocr_color not in self.config.SUPPORTED_OCR_COLORS:
                return ValidationResult(
                    False,
                    f"Invalid color: {ocr_color}. Supported colors: {', '.join(self.config.SUPPORTED_OCR_COLORS)}"
                )
        
        # Validate Fine-grained OCR (Box)
        if task == "Fine-grained OCR (Box)":
            if not ocr_box:
                return ValidationResult(False, "Bounding box coordinates are required for Fine-grained OCR (Box)")
            
            # Validate box format
            box_validation = self._validate_box_format(ocr_box)
            if not box_validation.is_valid:
                return box_validation
        
        # Validate formatted tasks
        if task in ["Format Text OCR", "Multi-crop OCR", "Multi-page OCR"]:
            if ocr_type and ocr_type not in self.config.SUPPORTED_OCR_TYPES:
                return ValidationResult(
                    False,
                    f"Invalid OCR type: {ocr_type}. Supported types: {', '.join(self.config.SUPPORTED_OCR_TYPES)}"
                )
        
        return ValidationResult(True)
    
    def _validate_box_format(self, ocr_box: str) -> ValidationResult:
        """Validate bounding box format"""
        try:
            # Remove brackets and split by comma
            coordinates = ocr_box.strip('[]').split(',')
            
            if len(coordinates) != 4:
                return ValidationResult(False, "Bounding box must contain exactly 4 coordinates")
            
            # Validate that all coordinates are integers
            coords = [int(coord.strip()) for coord in coordinates]
            
            # Validate coordinate ranges (basic validation)
            if any(coord < 0 for coord in coords):
                return ValidationResult(False, "Bounding box coordinates must be non-negative")
            
            # Validate that box makes geometric sense
            x1, y1, x2, y2 = coords
            if x2 <= x1 or y2 <= y1:
                return ValidationResult(False, "Invalid bounding box: x2 > x1 and y2 > y1 required")
            
            return ValidationResult(True)
            
        except (ValueError, AttributeError) as e:
            return ValidationResult(
                False,
                f"Invalid bounding box format: {ocr_box}. Expected format: [x1,y1,x2,y2]"
            )
    
    def _validate_images(self, images: List[UploadFile]) -> ValidationResult:
        """Validate uploaded images"""
        if not images:
            return ValidationResult(False, "At least one image is required")
        
        for img in images:
            # Validate file type
            if not img.content_type or not img.content_type.startswith("image/"):
                return ValidationResult(
                    False,
                    f"Invalid file type: {img.filename}. Only image files are supported"
                )
            
            # Validate supported formats
            if img.content_type not in self.config.SUPPORTED_FORMATS:
                return ValidationResult(
                    False,
                    f"Unsupported image format: {img.content_type}. "
                    f"Supported formats: {', '.join(self.config.SUPPORTED_FORMATS)}"
                )
            
            # Validate file size (if available)
            if hasattr(img, 'size') and img.size and img.size > self.config.MAX_FILE_SIZE:
                max_size_mb = self.config.MAX_FILE_SIZE / (1024 * 1024)
                return ValidationResult(
                    False,
                    f"File too large: {img.filename}. Maximum size: {max_size_mb}MB"
                )
        
        return ValidationResult(True)