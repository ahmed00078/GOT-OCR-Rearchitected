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

from models.ocr_model import OCRModelManager
from services.reasoning_service import SmolLM2ReasoningService, ExtractionType, ExtractionResult
from services.ocr_service import OCRService  # Service de base
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
        
        if self.reasoning_enabled:
            self.reasoning_service = SmolLM2ReasoningService(config)
    
    async def initialize_reasoning(self):
        """Initialiser le service de raisonnement de manière asynchrone"""
        if self.reasoning_enabled and self.reasoning_service:
            try:
                logger.info("Initialisation de SmolLM2...")
                await self.reasoning_service.load_model(
                    use_quantization=self.config.ENABLE_QUANTIZATION
                )
                logger.info("SmolLM2 initialisé avec succès")
            except Exception as e:
                logger.error(f"Erreur initialisation SmolLM2: {str(e)}")
                self.reasoning_enabled = False
    
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
            # === ÉTAPE 1: OCR STANDARD ===
            logger.info("Phase 1: Extraction OCR...")
            ocr_start = time.time()
            
            ocr_result = await super().process_images(
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
                logger.warning("SmolLM2 non chargé, tentative d'initialisation...")
                await self.initialize_reasoning()
                
                if not self.reasoning_service.is_loaded:
                    logger.error("Impossible de charger SmolLM2")
                    ocr_result.update({
                        "reasoning_enabled": False,
                        "extraction_error": "SmolLM2 loading failed",
                        "total_processing_time": time.time() - start_time
                    })
                    return ocr_result
            
            # === ÉTAPE 3: RAISONNEMENT IA ===
            logger.info("Phase 2: Raisonnement IA...")
            reasoning_start = time.time()
            
            # Validation du type d'extraction
            if not self.config.validate_extraction_type(extraction_type):
                raise ValueError(f"Type d'extraction invalide: {extraction_type}")
            
            # Extraction d'informations avec SmolLM2
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
            
            # === ÉTAPE 5: POST-PROCESSING ===
            # Enrichir avec des métriques de qualité
            enhanced_result["quality_metrics"] = self._calculate_quality_metrics(
                ocr_result, extraction_result
            )
            
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
        Optimisé pour SmolLM2:1.7B
        """
        if len(files_batch) != len(extraction_configs):
            raise ValueError("Le nombre de batches doit correspondre au nombre de configs")
        
        logger.info(f"Traitement batch: {len(files_batch)} documents")
        
        # Traitement séquentiel pour éviter la surcharge mémoire avec SmolLM2
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
    
    def _calculate_quality_metrics(
        self, 
        ocr_result: Dict[str, Any], 
        extraction_result: ExtractionResult
    ) -> Dict[str, Any]:
        """Calculer des métriques de qualité combinées"""
        
        ocr_text = ocr_result.get("text", "")
        extracted_data = extraction_result.extracted_data
        
        # Métriques de base
        text_length = len(ocr_text)
        word_count = len(ocr_text.split())
        
        # Métriques d'extraction
        extraction_completeness = extraction_result.confidence
        matches_ratio = len(extraction_result.raw_matches) / max(word_count, 1)
        
        # Score de qualité composite
        quality_score = (
            min(text_length / 100, 1.0) * 0.2 +  # Longueur du texte
            extraction_completeness * 0.6 +       # Complétude extraction
            min(matches_ratio * 10, 1.0) * 0.2    # Ratio de correspondances
        )
        
        return {
            "text_length": text_length,
            "word_count": word_count,
            "extraction_confidence": extraction_completeness,
            "matches_found": len(extraction_result.raw_matches),
            "matches_ratio": matches_ratio,
            "overall_quality_score": quality_score,
            "quality_grade": self._get_quality_grade(quality_score)
        }
    
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
            "carbon_footprint": {
                "name": "Bilan Carbone",
                "description": "Extraction d'émissions CO2, consommation énergétique",
                "typical_fields": ["carbon_emissions", "energy_consumption", "certification"],
                "use_cases": ["Produits électroniques", "Rapports environnementaux"]
            },
            "technical_specs": {
                "name": "Spécifications Techniques", 
                "description": "Caractéristiques produit, dimensions, performance",
                "typical_fields": ["product_name", "model", "dimensions", "power"],
                "use_cases": ["Fiches techniques", "Catalogues produits"]
            },
            "financial_data": {
                "name": "Données Financières",
                "description": "Prix, coûts, métriques financières",
                "typical_fields": ["prices", "costs", "revenue", "currencies"],
                "use_cases": ["Factures", "Rapports financiers", "Devis"]
            },
            "contact_info": {
                "name": "Informations de Contact",
                "description": "Noms, emails, téléphones, adresses",
                "typical_fields": ["names", "emails", "phones", "addresses"],
                "use_cases": ["Cartes de visite", "Annuaires", "Documents RH"]
            },
            "custom": {
                "name": "Extraction Personnalisée",
                "description": "Extraction selon vos instructions spécifiques",
                "typical_fields": ["Variable selon instructions"],
                "use_cases": ["Cas d'usage spécialisés", "Données métier spécifiques"]
            }
        }
    
    async def cleanup(self):
        """Nettoyer toutes les ressources"""
        logger.info("Nettoyage Enhanced OCR Service...")
        
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