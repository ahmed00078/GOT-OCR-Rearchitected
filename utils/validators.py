# File: utils/validators.py - Version Multi-Page
"""
Request validators for GOT-OCR 2.0 API - Support Multi-Page
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
    """Validator for OCR processing requests - Enhanced for PDF support"""
    
    def __init__(self):
        self.config = Config()
        # === NOUVELLES CONSTANTES POUR PDF ===
        self.max_pdf_pages = 50
        self.max_pdf_size_mb = 100  # 100MB max pour les PDFs
    
    def validate_request(
        self,
        task: str,
        ocr_type: Optional[str],
        ocr_box: Optional[str],
        ocr_color: Optional[str],
        images: List[UploadFile]
    ) -> ValidationResult:
        """Validate complete OCR request - Enhanced for PDF"""
        
        # Validate task
        task_validation = self._validate_task(task)
        if not task_validation.is_valid:
            return task_validation
        
        # Validate task-specific parameters
        params_validation = self._validate_task_parameters(task, ocr_type, ocr_box, ocr_color)
        if not params_validation.is_valid:
            return params_validation
        
        # Validate files (images + PDFs)
        files_validation = self._validate_files(images)
        if not files_validation.is_valid:
            return files_validation
        
        # === VALIDATION SPÉCIFIQUE MULTI-PAGE ===
        if task == "Multi-page OCR":
            multipage_validation = self._validate_multipage_task(images)
            if not multipage_validation.is_valid:
                return multipage_validation
        
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
    
    # === NOUVELLE MÉTHODE : VALIDATION DES FICHIERS (IMAGES + PDF) ===
    def _validate_files(self, files: List[UploadFile]) -> ValidationResult:
        """Validate uploaded files - Enhanced for PDF support"""
        if not files:
            return ValidationResult(False, "At least one file is required")
        
        pdf_count = 0
        image_count = 0
        
        for file in files:
            # Validate file type
            if not file.content_type:
                return ValidationResult(
                    False,
                    f"Unknown file type: {file.filename}"
                )
            
            # === SUPPORT PDF ===
            if file.content_type == "application/pdf":
                pdf_count += 1
                # Validation spécifique PDF
                pdf_validation = self._validate_pdf_file(file)
                if not pdf_validation.is_valid:
                    return pdf_validation
                    
            # === SUPPORT IMAGES ===
            elif file.content_type.startswith("image/"):
                image_count += 1
                # Validation spécifique images
                image_validation = self._validate_image_file(file)
                if not image_validation.is_valid:
                    return image_validation
                    
            else:
                return ValidationResult(
                    False,
                    f"Unsupported file type: {file.content_type}. "
                    f"Supported: images (JPEG, PNG, TIFF) and PDF files"
                )
        
        # === VALIDATION LOGIQUE MÉTIER ===
        # Si multi-page OCR et mélange PDF/images, avertir
        if pdf_count > 0 and image_count > 0:
            logger.warning(f"Mixed upload: {pdf_count} PDFs and {image_count} images")
        
        # Limite globale de fichiers
        if len(files) > 20:  # Limite raisonnable
            return ValidationResult(False, "Too many files. Maximum 20 files per request")
        
        return ValidationResult(True)
    
    # === NOUVELLE MÉTHODE : VALIDATION SPÉCIFIQUE PDF ===
    def _validate_pdf_file(self, pdf_file: UploadFile) -> ValidationResult:
        """Validate PDF file specifically"""
        
        # Validation de la taille (approximative via headers si disponible)
        if hasattr(pdf_file, 'size') and pdf_file.size:
            max_size_bytes = self.max_pdf_size_mb * 1024 * 1024
            if pdf_file.size > max_size_bytes:
                return ValidationResult(
                    False,
                    f"PDF too large: {pdf_file.filename}. Maximum size: {self.max_pdf_size_mb}MB"
                )
        
        # Validation de l'extension
        if not pdf_file.filename.lower().endswith('.pdf'):
            return ValidationResult(
                False,
                f"Invalid PDF file: {pdf_file.filename}. Must have .pdf extension"
            )
        
        return ValidationResult(True)
    
    # === MÉTHODE REFACTORISÉE : VALIDATION SPÉCIFIQUE IMAGES ===
    def _validate_image_file(self, image_file: UploadFile) -> ValidationResult:
        """Validate image file specifically"""
        
        # Validate supported formats
        if image_file.content_type not in self.config.SUPPORTED_FORMATS:
            return ValidationResult(
                False,
                f"Unsupported image format: {image_file.content_type}. "
                f"Supported formats: {', '.join(self.config.SUPPORTED_FORMATS)}"
            )
        
        # Validate file size (if available)
        if hasattr(image_file, 'size') and image_file.size and image_file.size > self.config.MAX_FILE_SIZE:
            max_size_mb = self.config.MAX_FILE_SIZE / (1024 * 1024)
            return ValidationResult(
                False,
                f"Image too large: {image_file.filename}. Maximum size: {max_size_mb}MB"
            )
        
        return ValidationResult(True)
    
    # === NOUVELLE MÉTHODE : VALIDATION MULTI-PAGE ===
    def _validate_multipage_task(self, files: List[UploadFile]) -> ValidationResult:
        """Validate multi-page OCR specific requirements"""
        
        # Pour multi-page, préférer un seul PDF ou plusieurs images ordonnées
        pdf_files = [f for f in files if f.content_type == "application/pdf"]
        image_files = [f for f in files if f.content_type.startswith("image/")]
        
        # if len(pdf_files) > 1:
        #     return ValidationResult(
        #         False,
        #         "Multi-page OCR supports only one PDF file at a time"
        #     )
        
        if len(pdf_files) == 1 and len(image_files) > 0:
            logger.warning("Multi-page OCR: mixing PDF and images may produce unexpected results")
        
        if len(pdf_files) == 0 and len(image_files) > 20:
            return ValidationResult(
                False,
                "Too many images for multi-page OCR. Maximum 20 images or use a single PDF"
            )
        
        return ValidationResult(True)
    
    # === NOUVELLE MÉTHODE UTILITAIRE ===
    def get_file_statistics(self, files: List[UploadFile]) -> dict:
        """Get statistics about uploaded files"""
        stats = {
            "total_files": len(files),
            "pdf_count": 0,
            "image_count": 0,
            "total_size_mb": 0,
            "file_types": []
        }
        
        for file in files:
            if file.content_type == "application/pdf":
                stats["pdf_count"] += 1
            elif file.content_type.startswith("image/"):
                stats["image_count"] += 1
            
            stats["file_types"].append(file.content_type)
            
            if hasattr(file, 'size') and file.size:
                stats["total_size_mb"] += file.size / (1024 * 1024)
        
        return stats