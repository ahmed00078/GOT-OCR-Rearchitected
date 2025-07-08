# File: services/enhanced_ocr_service.py
"""
Service OCR amélioré avec intégration SmolLM2
Pipeline complet : OCR → Text → Reasoning → Structured Data
"""

import logging
import asyncio
import time
from typing import List, Optional, Dict, Any, Union

from fastapi import UploadFile, BackgroundTasks
from PIL import Image
import cv2
import numpy as np
from paddleocr import LayoutDetection
import fitz

from models.ocr_model import OCRModelManager
from services.reasoning_service import SmolLM2ReasoningService, ExtractionType, ExtractionResult
from services.ollama_reasoning_service import OllamaReasoningService, ExtractionType, ExtractionResult
from services.ocr_service import OCRService
from config import Config

logger = logging.getLogger(__name__)


class EnhancedOCRService(OCRService):
    """
    Service OCR amélioré avec capacités de raisonnement IA
    Hérite du service OCR de base et ajoute SmolLM2
    """
    
    def __init__(self, model_manager: OCRModelManager, config: Config):
        # Initialiser le service OCR de base
        super().__init__(model_manager, config)
        
        # Ajouter le service de raisonnement
        self.reasoning_service: Optional[SmolLM2ReasoningService] = None
        self.reasoning_enabled = config.reasoning_enabled
        
        # Ajouter la segmentation layout
        self.layout_model: Optional[LayoutDetection] = None
        self.layout_enabled = getattr(config, 'LAYOUT_ENABLED', True)
        
        if self.reasoning_enabled:
            self.reasoning_service = SmolLM2ReasoningService(config)
    
    async def initialize_reasoning(self):
        """Initialiser le service de raisonnement de manière asynchrone"""
        if self.reasoning_enabled and self.reasoning_service:
            try:
                logger.info("Initialisation d'Ollama...")
                await self.reasoning_service.load_model(
                    use_quantization=self.config.ENABLE_QUANTIZATION
                )
                logger.info("Ollama initialisé avec succès")
            except Exception as e:
                logger.error(f"Erreur initialisation Ollama: {str(e)}")
                self.reasoning_enabled = False
        
        # Initialiser le modèle de layout
        if self.layout_enabled:
            try:
                logger.info("Initialisation PP-DocLayout...")
                self.layout_model = LayoutDetection(model_name="PP-DocLayout_plus-L")
                logger.info("PP-DocLayout initialisé avec succès")
            except Exception as e:
                logger.error(f"Erreur initialisation PP-DocLayout: {str(e)}")
                self.layout_enabled = False
    
    async def process_with_reasoning(
        self,
        task: str,
        images: List[UploadFile],
        background_tasks: BackgroundTasks,
        extraction_type: str,
        custom_instructions: Optional[str] = None,
        ocr_type: Optional[str] = None,
        ocr_box: Optional[str] = None,
        ocr_color: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Pipeline complet : OCR + Reasoning
        
        Args:
            task: Type de tâche OCR
            images: Fichiers uploadés
            background_tasks: Tâches en arrière-plan
            extraction_type: Type d'extraction d'informations
            custom_instructions: Instructions personnalisées
            Autres: Paramètres OCR standards
            
        Returns:
            Résultat combiné OCR + raisonnement
        """
        logger.info(f"Démarrage pipeline OCR + Reasoning : {task} → {extraction_type}")
        start_time = time.time()
        
        try:
            # === ÉTAPE 1: OCR AVEC SEGMENTATION ===
            logger.info("Phase 1: Extraction OCR avec segmentation...")
            ocr_start = time.time()
            
            ocr_result = await self._process_with_layout(
                task=task,
                images=images,
                background_tasks=background_tasks,
                ocr_type=ocr_type,
                ocr_box=ocr_box,
                ocr_color=ocr_color
            )
            
            ocr_time = time.time() - ocr_start
            logger.info(f"OCR terminé en {ocr_time:.2f}s, {len(ocr_result['text'])} caractères extraits")
            
            # === ÉTAPE 2: VÉRIFICATION DU REASONING ===
            if not self.reasoning_enabled or not self.reasoning_service:
                logger.warning("Reasoning désactivé, retour du résultat OCR seul")
                ocr_result.update({
                    "reasoning_enabled": False,
                    "extraction_result": None,
                    "total_processing_time": time.time() - start_time
                })
                return ocr_result
            
            if not self.reasoning_service.is_loaded:
                logger.warning("Ollama non chargé, tentative d'initialisation...")
                await self.initialize_reasoning()
                
                if not self.reasoning_service.is_loaded:
                    logger.error("Impossible de charger Ollama")
                    ocr_result.update({
                        "reasoning_enabled": False,
                        "extraction_error": "Ollama loading failed",
                        "total_processing_time": time.time() - start_time
                    })
                    return ocr_result
            
            # === ÉTAPE 3: RAISONNEMENT IA ===
            logger.info("Phase 2: Raisonnement IA...")
            reasoning_start = time.time()
            
            # Validation du type d'extraction
            if not self.config.validate_extraction_type(extraction_type):
                raise ValueError(f"Type d'extraction invalide: {extraction_type}")
            
            # Extraction d'informations avec Ollama
            extraction_result = await self.reasoning_service.extract_information(
                text=ocr_result["text"],
                extraction_type=ExtractionType(extraction_type),
                custom_instructions=custom_instructions,
                max_length=self.config.REASONING_CONFIG["max_context_length"]
            )
            
            reasoning_time = time.time() - reasoning_start
            total_time = time.time() - start_time
            
            logger.info(f"Reasoning terminé en {reasoning_time:.2f}s, confiance: {extraction_result.confidence:.3f}")
            
            # === ÉTAPE 4: RÉSULTAT ENRICHI ===
            enhanced_result = {
                **ocr_result,  # Résultat OCR de base
                
                # Nouvelles données de raisonnement
                "reasoning_enabled": True,
                "extraction_type": extraction_type,
                "extraction_result": {
                    "confidence": extraction_result.confidence,
                    "extracted_data": extraction_result.extracted_data,
                    "raw_matches": extraction_result.raw_matches,
                    "processing_time": extraction_result.processing_time
                },
                
                # Métriques de performance
                "performance_metrics": {
                    "ocr_time": ocr_time,
                    "reasoning_time": reasoning_time,
                    "total_time": total_time,
                    "text_length": len(ocr_result["text"]),
                    "confidence_score": extraction_result.confidence
                },
                
                # Métadonnées du modèle
                "models_used": {
                    "ocr_model": self.model_manager.config.MODEL_NAME,
                    "reasoning_model": extraction_result.model_used
                }
            }
            
            logger.info(f"Pipeline complet terminé en {total_time:.2f}s")
            return enhanced_result
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Erreur pipeline OCR+Reasoning: {str(e)}")
            
            # Retourner au moins le résultat OCR en cas d'erreur
            try:
                fallback_result = await super().process_images(
                    task=task,
                    images=images,
                    background_tasks=background_tasks,
                    ocr_type=ocr_type,
                    ocr_box=ocr_box,
                    ocr_color=ocr_color
                )
                fallback_result.update({
                    "reasoning_enabled": False,
                    "reasoning_error": str(e),
                    "total_processing_time": total_time
                })
                return fallback_result
            except Exception as fallback_error:
                logger.error(f"Erreur fallback OCR: {str(fallback_error)}")
                raise
    
    async def batch_process_with_reasoning(
        self,
        files_batch: List[List[UploadFile]],
        extraction_configs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Traitement par batch pour plusieurs documents
        Optimisé pour Ollama
        """
        if len(files_batch) != len(extraction_configs):
            raise ValueError("Le nombre de batches doit correspondre au nombre de configs")
        
        logger.info(f"Traitement batch: {len(files_batch)} documents")
        
        # Traitement séquentiel pour éviter la surcharge avec Ollama
        results = []
        
        for i, (files, config) in enumerate(zip(files_batch, extraction_configs)):
            logger.info(f"Traitement document {i+1}/{len(files_batch)}")
            
            try:
                # Créer des tâches d'arrière-plan temporaires
                from fastapi import BackgroundTasks
                bg_tasks = BackgroundTasks()
                
                result = await self.process_with_reasoning(
                    task=config.get("task", "Multi-page OCR"),
                    images=files,
                    background_tasks=bg_tasks,
                    extraction_type=config.get("extraction_type", "technical_specs"),
                    custom_instructions=config.get("custom_instructions"),
                    ocr_type=config.get("ocr_type"),
                    ocr_box=config.get("ocr_box"),
                    ocr_color=config.get("ocr_color")
                )
                
                results.append(result)
                
                # Petit délai entre documents pour éviter la surchauffe
                if i < len(files_batch) - 1:
                    await asyncio.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"Erreur traitement document {i+1}: {str(e)}")
                results.append({
                    "error": str(e),
                    "document_index": i,
                    "reasoning_enabled": False
                })
        
        logger.info(f"Batch terminé: {len(results)} résultats")
        return results

    def _get_quality_grade(self, score: float) -> str:
        """Convertir le score en note qualitative"""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.7:
            return "Bon"
        elif score >= 0.5:
            return "Correct"
        elif score >= 0.3:
            return "Faible"
        else:
            return "Très faible"
    
    async def get_supported_extractions(self) -> Dict[str, Dict[str, Any]]:
        """Obtenir la liste des types d'extraction supportés avec descriptions"""
        return {
            "custom": {
                "name": "Custom Extraction",
                "description": "AI-powered data extraction with custom instructions",
                "typical_fields": ["Variable based on instructions"],
                "use_cases": [
                    "Product specifications and prices",
                    "Contact information extraction", 
                    "Carbon footprint and environmental data",
                    "Financial metrics and costs",
                    "Technical specifications",
                    "Any custom data extraction task"
                ],
                "examples": [
                    "Extract all product names, prices, and warranty information",
                    "Find contact information including emails, phones, and addresses",
                    "Extract carbon footprint data and environmental certifications",
                    "Get technical specifications, dimensions, and performance metrics"
                ]
            }
        }
    
    async def _process_with_layout(self, task: str, images: List[UploadFile], background_tasks: BackgroundTasks, ocr_type: Optional[str] = None, ocr_box: Optional[str] = None, ocr_color: Optional[str] = None) -> Dict[str, Any]:
        """Traiter les images avec segmentation layout si activée"""
        if not self.layout_enabled or not self.layout_model:
            return await super().process_images(task, images, background_tasks, ocr_type, ocr_box, ocr_color)
        
        logger.info("Traitement avec segmentation PP-DocLayout")
        
        try:
            # Charger images manuellement pour éviter les problèmes async
            loaded_images = await self._load_images_for_segmentation(images)
            all_results = []
            
            for i, (image, filename) in enumerate(loaded_images):
                logger.info(f"Segmentation image {i+1}/{len(loaded_images)}: {filename}")
                
                # Convertir PIL vers OpenCV
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Analyse layout
                output = self.layout_model.predict(cv_image, batch_size=1, layout_nms=True)
                boxes = output[0]['boxes']
                
                logger.info(f"🔍 Zones détectées: {len(boxes)}")
                print(f"🔍 Zones détectées: {len(boxes)}")  # Force l'affichage
                
                # Afficher le détail des zones détectées
                if boxes:
                    zone_types = {}
                    for box in boxes:
                        label = box['label']
                        zone_types[label] = zone_types.get(label, 0) + 1
                    
                    zone_summary = ", ".join([f"{label}: {count}" for label, count in zone_types.items()])
                    logger.info(f"📋 Types de zones: {zone_summary}")
                    print(f"📋 Types de zones: {zone_summary}")  # Force l'affichage
                
                if not boxes:
                    text = await self._extract_text_from_image(image, filename)
                    all_results.append({"text": text, "images": [], "filename": filename})
                    continue
                
                # Trier les zones par ordre de lecture
                boxes_sorted = sorted(boxes, key=lambda box: (box['coordinate'][1], box['coordinate'][0]))
                
                # Traiter chaque zone
                markdown_parts = []
                images_found = []
                
                for j, box in enumerate(boxes_sorted):
                    x1, y1, x2, y2 = map(int, box['coordinate'])
                    label = box['label']
                    score = box.get('score', 0.0)
                    
                    logger.info(f"  📍 Zone {j+1}: {label} (conf: {score:.2f}) - [{x1},{y1},{x2},{y2}]")
                    print(f"  📍 Zone {j+1}: {label} (conf: {score:.2f}) - [{x1},{y1},{x2},{y2}]")  # Force l'affichage
                    
                    # Découper la zone
                    cropped_cv = cv_image[y1:y2, x1:x2]
                    cropped_pil = Image.fromarray(cv2.cvtColor(cropped_cv, cv2.COLOR_BGR2RGB))
                    
                    if label.lower() in ['figure', 'image', 'chart']:
                        # Encoder l'image en base64
                        import base64
                        import io
                        buffer = io.BytesIO()
                        cropped_pil.save(buffer, format='PNG')
                        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        
                        image_info = {
                            "id": f"img-{i}-{j}",
                            "top_left_x": x1,
                            "top_left_y": y1,
                            "bottom_right_x": x2,
                            "bottom_right_y": y2,
                            "image_base64": img_base64
                        }
                        images_found.append(image_info)
                        markdown_parts.append(f"![{image_info['id']}]({image_info['id']})")
                    else:
                        # Extraire le texte de la zone
                        text = await self._extract_text_from_image(cropped_pil, f"{filename}_zone_{j}")
                        if text.strip():
                            markdown_parts.append(text)
                
                all_results.append({
                    "text": "\n\n".join(markdown_parts),
                    "images": images_found,
                    "filename": filename
                })
            
            # Combiner tous les résultats
            combined_text = "\n\n".join([r["text"] for r in all_results if r["text"]])
            combined_images = []
            for r in all_results:
                combined_images.extend(r["images"])
            
            return {
                "result_id": f"ocr_{int(time.time())}",
                "text": combined_text,
                "html_available": len(combined_text) > 0,
                "html": f"<div class='ocr-result'><pre>{combined_text}</pre></div>" if combined_text else None,
                "download_url": None,
                "multipage_info": {
                    "total_pages": len(all_results),
                    "total_images": len(combined_images),
                    "processing_method": "layout_segmentation"
                }
            }
            
        except Exception as e:
            logger.error(f"Erreur segmentation layout: {str(e)}, fallback vers OCR standard")
            return await super().process_images(task, images, background_tasks, ocr_type, ocr_box, ocr_color)
    
    async def _load_images_for_segmentation(self, images: List[UploadFile]):
        """Charger les images spécialement pour la segmentation"""
        loaded_images = []
        
        for upload_file in images:
            try:
                # Lire le contenu du fichier
                content = await upload_file.read()
                await upload_file.seek(0)  # Reset pour les autres usages
                
                filename = upload_file.filename or "unknown"
                
                # Vérifier si c'est un PDF
                if filename.lower().endswith('.pdf'):
                    # Convertir PDF en images
                    import fitz  # PyMuPDF
                    import io
                    
                    pdf_doc = fitz.open(stream=content, filetype="pdf")
                    for page_num in range(len(pdf_doc)):
                        page = pdf_doc[page_num]
                        # Convertir la page en image
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom pour meilleure qualité
                        img_data = pix.tobytes("png")
                        
                        # Convertir en PIL Image
                        pil_image = Image.open(io.BytesIO(img_data)).convert('RGB')
                        loaded_images.append((pil_image, f"{filename}_page_{page_num+1}"))
                    
                    pdf_doc.close()
                else:
                    # C'est une image normale
                    import io
                    pil_image = Image.open(io.BytesIO(content)).convert('RGB')
                    loaded_images.append((pil_image, filename))
                    
            except Exception as e:
                logger.error(f"Erreur chargement {upload_file.filename}: {str(e)}")
                continue
                
        return loaded_images
    
    async def _extract_text_from_image(self, pil_image: Image.Image, filename: str) -> str:
        """Extraire le texte d'une image PIL avec GOT-OCR"""
        try:
            import torch
            # Utiliser le modèle GOT-OCR directement
            inputs = self.model_manager.processor(pil_image, return_tensors="pt")
            inputs = {k: v.to(self.model_manager.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                generated_ids = self.model_manager.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=1024,
                    tokenizer=self.model_manager.processor.tokenizer,
                    stop_strings="<|im_end|>",
                )
            
            result = self.model_manager.processor.decode(
                generated_ids[0, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            
            return result.strip()
            
        except Exception as e:
            logger.error(f"Erreur extraction texte pour {filename}: {str(e)}")
            return ""
    
    async def cleanup(self):
        """Nettoyer toutes les ressources"""
        logger.info("Nettoyage Enhanced OCR Service...")
        
        # Nettoyer le modèle de layout
        if self.layout_model:
            self.layout_model = None
            logger.info("Modèle layout nettoyé")
        
        # Nettoyer le service de raisonnement
        if self.reasoning_service:
            await self.reasoning_service.cleanup()
        
        # Nettoyer le service OCR de base (via super())
        # Note: la classe parent n'a pas de méthode cleanup, mais on peut l'ajouter si nécessaire
        
        logger.info("Nettoyage Enhanced OCR Service terminé")
    
    @property
    def is_reasoning_available(self) -> bool:
        """Vérifier si le raisonnement est disponible"""
        return (
            self.reasoning_enabled and 
            self.reasoning_service is not None and 
            self.reasoning_service.is_loaded
        )