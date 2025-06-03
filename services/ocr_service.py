# File: services/ocr_service.py - Version Multi-Page Enhanced
"""
OCR Service for GOT-OCR 2.0 - Enhanced Multi-Page Support
Handles OCR processing logic and file management including PDF conversion
"""

import logging
import os
import shutil
import tempfile
import uuid
import base64
from typing import List, Optional, Dict, Any, Tuple

from fastapi import UploadFile, BackgroundTasks
from PIL import Image

from models.ocr_model import OCRModelManager
from config import Config
from render import render_ocr_text
from services.pdf_service import PDFService

logger = logging.getLogger(__name__)


class OCRService:
    """Service class for OCR processing operations - Enhanced for Multi-Page"""
    
    def __init__(self, model_manager: OCRModelManager, config: Config):
        self.model_manager = model_manager
        self.config = config
        self.pdf_service = PDFService(config)
    
    async def process_images(
        self,
        task: str,
        images: List[UploadFile],
        background_tasks: BackgroundTasks,
        ocr_type: Optional[str] = None,
        ocr_box: Optional[str] = None,
        ocr_color: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process uploaded files with OCR - Enhanced for Multi-Page Support
        
        === NOUVEAUTÉS ===
        - Détection automatique PDF vs Images
        - Conversion automatique PDF -> Images
        - Traitement par batch pour performance
        - Concaténation intelligente des résultats multi-pages
        """
        
        # Create temporary workspace
        temp_dir = tempfile.mkdtemp()
        background_tasks.add_task(self._cleanup_tempdir, temp_dir)
        
        try:
            # === ÉTAPE 1: ANALYSER ET PRÉPARER LES FICHIERS ===
            file_analysis = await self._analyze_uploaded_files(images)
            logger.info(f"File analysis: {file_analysis}")
            
            # === ÉTAPE 2: CONVERTIR LES PDFs EN IMAGES ===
            all_image_paths = await self._prepare_all_images(images, temp_dir, file_analysis)
            logger.info(f"Prepared {len(all_image_paths)} images for processing")
            
            # === ÉTAPE 3: CHARGER TOUTES LES IMAGES ===
            pil_images = await self._load_images_optimized(all_image_paths)
            
            # === ÉTAPE 4: TRAITEMENT OCR ===
            if task == "Multi-page OCR" and len(pil_images) > 1:
                # Traitement spécialisé multi-page
                text_result = await self._process_multipage_ocr(
                    pil_images, task, ocr_type, ocr_box, ocr_color, file_analysis
                )
            else:
                # Traitement standard
                task_config = self._prepare_task_config(task, ocr_type, ocr_box, ocr_color)
                text_result = self.model_manager.generate_text(pil_images, task_config)
            
            # === ÉTAPE 5: GÉNÉRATION DU RÉSULTAT ===
            result_id = str(uuid.uuid4())
            
            # Handle rendering for formatted outputs
            html_content = None
            if self._requires_html_rendering(task):
                html_content = await self._render_html_output(
                    text_result, temp_dir, result_id, task_config if 'task_config' in locals() else {"task": task}
                )
            
            # Prepare response with multi-page metadata
            response = {
                "result_id": result_id,
                "text": text_result,
                "html_available": html_content is not None,
                # === MÉTADONNÉES MULTI-PAGE ===
                "multipage_info": {
                    "total_pages": len(pil_images),
                    "pdf_count": file_analysis["pdf_count"],
                    "image_count": file_analysis["image_count"],
                    "processing_method": "multipage" if task == "Multi-page OCR" else "standard"
                }
            }
            
            if html_content:
                encoded_html = base64.b64encode(html_content.encode()).decode()
                response.update({
                    "html": encoded_html,
                    "download_url": f"/results/{result_id}"
                })
            
            logger.info(f"Successfully processed {len(images)} files ({len(pil_images)} pages) for task: {task}")
            return response
            
        except Exception as e:
            logger.error(f"OCR processing failed: {str(e)}")
            raise
    
    # === NOUVELLE MÉTHODE: ANALYSE DES FICHIERS UPLOADÉS ===
    async def _analyze_uploaded_files(self, files: List[UploadFile]) -> Dict[str, Any]:
        """Analyze uploaded files to determine processing strategy"""
        analysis = {
            "total_files": len(files),
            "pdf_count": 0,
            "image_count": 0,
            "file_types": [],
            "estimated_pages": 0,
            "has_mixed_types": False
        }
        
        for file in files:
            if file.content_type == "application/pdf":
                analysis["pdf_count"] += 1
                # Estimation grossière: 1 PDF = ~10 pages en moyenne
                analysis["estimated_pages"] += 10
            elif file.content_type.startswith("image/"):
                analysis["image_count"] += 1
                analysis["estimated_pages"] += 1
            
            analysis["file_types"].append(file.content_type)
        
        analysis["has_mixed_types"] = analysis["pdf_count"] > 0 and analysis["image_count"] > 0
        return analysis
    
    # === NOUVELLE MÉTHODE: PRÉPARATION UNIFIÉE DES IMAGES ===
    async def _prepare_all_images(
        self, 
        files: List[UploadFile], 
        temp_dir: str, 
        file_analysis: Dict[str, Any]
    ) -> List[str]:
        """
        Prepare all images from mixed PDF and image uploads
        Returns list of image paths ready for OCR processing
        """
        all_image_paths = []
        
        for file in files:
            if file.content_type == "application/pdf":
                # === CONVERSION PDF -> IMAGES ===
                logger.info(f"Converting PDF: {file.filename}")
                pdf_images = await self.pdf_service.convert_pdf_to_images(
                    file, 
                    temp_dir,
                    dpi=self.config.PDF_CONVERSION_DPI
                )
                all_image_paths.extend(pdf_images)
                logger.info(f"PDF {file.filename} converted to {len(pdf_images)} pages")
                
            elif file.content_type.startswith("image/"):
                # === SAUVEGARDE D'IMAGES ===
                image_path = await self._save_single_image(file, temp_dir)
                all_image_paths.append(image_path)
                logger.debug(f"Image saved: {file.filename}")
        
        return all_image_paths
    
    # === NOUVELLE MÉTHODE: SAUVEGARDE D'UNE IMAGE ===
    async def _save_single_image(self, image_file: UploadFile, temp_dir: str) -> str:
        """Save a single image file"""
        if not image_file.content_type.startswith("image/"):
            raise ValueError(f"Invalid file type: {image_file.content_type}")
        
        # Generate unique filename to avoid conflicts
        base_name = os.path.splitext(image_file.filename)[0]
        extension = os.path.splitext(image_file.filename)[1]
        unique_name = f"{base_name}_{uuid.uuid4().hex[:8]}{extension}"
        
        img_path = os.path.join(temp_dir, unique_name)
        with open(img_path, "wb") as buffer:
            content = await image_file.read()
            buffer.write(content)
        
        return img_path
    
    # === NOUVELLE MÉTHODE: CHARGEMENT OPTIMISÉ DES IMAGES ===
    async def _load_images_optimized(self, image_paths: List[str]) -> List[Image.Image]:
        """Load images with memory optimization for large batches"""
        pil_images = []
        
        for img_path in image_paths:
            try:
                # Ouvrir l'image avec optimisation mémoire
                image = Image.open(img_path)
                
                # Convertir en RGB si nécessaire (pour PDFs convertis)
                if image.mode not in ['RGB', 'L']:
                    image = image.convert('RGB')
                
                # Optionnel: redimensionner si l'image est trop grande
                if image.size[0] > 4000 or image.size[1] > 4000:
                    logger.warning(f"Large image detected: {image.size}, consider resizing")
                
                pil_images.append(image)
                logger.debug(f"Loaded image: {os.path.basename(img_path)} ({image.size})")
                
            except Exception as e:
                logger.error(f"Failed to load image {img_path}: {str(e)}")
                raise ValueError(f"Cannot load image: {os.path.basename(img_path)}")
        
        return pil_images
    
    # === NOUVELLE MÉTHODE: TRAITEMENT SPÉCIALISÉ MULTI-PAGE ===
    async def _process_multipage_ocr(
        self,
        pil_images: List[Image.Image],
        task: str,
        ocr_type: Optional[str],
        ocr_box: Optional[str],
        ocr_color: Optional[str],
        file_analysis: Dict[str, Any]
    ) -> str:
        """
        Specialized processing for multi-page documents
        Handles batch processing and intelligent result concatenation
        """
        logger.info(f"Starting multi-page OCR for {len(pil_images)} pages")
        
        # Préparer la configuration de base
        base_config = self._prepare_task_config(task, ocr_type, ocr_box, ocr_color)
        
        # Traitement par batch pour optimiser la mémoire
        batch_size = self.config.MULTIPAGE_BATCH_SIZE
        all_results = []
        
        for i in range(0, len(pil_images), batch_size):
            batch_images = pil_images[i:i + batch_size]
            batch_start = i + 1
            batch_end = min(i + batch_size, len(pil_images))
            
            logger.info(f"Processing batch {batch_start}-{batch_end}")
            
            try:
                # Traitement du batch
                batch_result = self.model_manager.generate_text(batch_images, base_config)
                
                # Si c'est un batch de plusieurs pages, séparer intelligemment
                if len(batch_images) > 1:
                    # Tentative de séparation par page (heuristique simple)
                    page_results = self._split_batch_result(batch_result, len(batch_images))
                else:
                    page_results = [batch_result]
                
                # Ajouter avec numérotation des pages
                for j, page_result in enumerate(page_results):
                    page_num = batch_start + j
                    formatted_result = self.config.MULTIPAGE_CONCAT_SEPARATOR.format(page_num)
                    formatted_result += page_result.strip()
                    all_results.append(formatted_result)
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_start}-{batch_end}: {str(e)}")
                # Ajouter une erreur dans le résultat plutôt que de planter
                error_msg = f"[ERROR: Failed to process pages {batch_start}-{batch_end}: {str(e)}]"
                all_results.append(error_msg)
        
        # Concaténer tous les résultats
        final_result = "\n".join(all_results)
        
        # Ajouter un résumé en en-tête
        summary = f"=== MULTI-PAGE OCR RESULTS ===\n"
        summary += f"Total pages processed: {len(pil_images)}\n"
        summary += f"PDFs: {file_analysis['pdf_count']}, Images: {file_analysis['image_count']}\n"
        summary += f"Processing date: {uuid.uuid4().hex[:8]}\n"
        summary += "=" * 50 + "\n\n"
        
        return summary + final_result
    
    # === MÉTHODE UTILITAIRE: SÉPARATION DES RÉSULTATS PAR BATCH ===
    def _split_batch_result(self, batch_result: str, num_pages: int) -> List[str]:
        """
        Try to intelligently split batch OCR result into individual pages
        This is a heuristic approach - may not be perfect
        """
        if num_pages == 1:
            return [batch_result]
        
        # Stratégies de séparation (par ordre de priorité)
        
        # 1. Chercher des marqueurs de page naturels
        page_markers = ['\f', '\n\n\n', '---', '===']
        for marker in page_markers:
            parts = batch_result.split(marker)
            if len(parts) == num_pages:
                return [part.strip() for part in parts]
        
        # 2. Séparation par longueur approximative
        avg_length = len(batch_result) // num_pages
        pages = []
        start = 0
        
        for i in range(num_pages - 1):
            # Chercher une coupure naturelle (fin de phrase/paragraphe) près de la position cible
            target_pos = start + avg_length
            best_cut = target_pos
            
            # Chercher un bon endroit pour couper (fin de ligne, point, etc.)
            for offset in range(-50, 51):
                pos = target_pos + offset
                if 0 <= pos < len(batch_result):
                    char = batch_result[pos]
                    if char in '\n.!?':
                        best_cut = pos + 1
                        break
            
            pages.append(batch_result[start:best_cut].strip())
            start = best_cut
        
        # Dernière page = tout ce qui reste
        pages.append(batch_result[start:].strip())
        
        return pages
    
    # === MÉTHODES EXISTANTES MAINTENUES ===
    
    async def _save_uploaded_images(
        self, 
        images: List[UploadFile], 
        temp_dir: str
    ) -> List[str]:
        """
        Legacy method maintained for compatibility
        Use _prepare_all_images for new multi-page functionality
        """
        image_paths = []
        
        for img in images:
            if img.content_type.startswith("image/"):
                img_path = await self._save_single_image(img, temp_dir)
                image_paths.append(img_path)
            else:
                logger.warning(f"Skipping non-image file in legacy method: {img.filename}")
        
        return image_paths
    
    def _prepare_task_config(
        self,
        task: str,
        ocr_type: Optional[str] = None,
        ocr_box: Optional[str] = None,
        ocr_color: Optional[str] = None
    ) -> Dict[str, Any]:
        """Prepare task configuration for model processing - Unchanged"""
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
        """Check if task requires HTML rendering - Unchanged"""
        return any(keyword in task for keyword in ["Format", "Fine-grained", "Multi"])
    
    async def _render_html_output(
        self,
        text_result: str,
        temp_dir: str,
        result_id: str,
        task_config: Dict[str, Any]
    ) -> Optional[str]:
        """Render OCR text to HTML format - Unchanged"""
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
        """Clean up temporary directory - Unchanged"""
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
            logger.debug(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temporary directory {temp_dir}: {str(e)}")