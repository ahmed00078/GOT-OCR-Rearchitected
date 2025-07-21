# File: services/reasoning_service.py
"""
Extraction d'informations structurées à partir du texte OCR
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    GenerationConfig
)

logger = logging.getLogger(__name__)


class ExtractionType(Enum):
    """Types d'extraction supportés"""
    CUSTOM = "custom"


@dataclass
class ExtractionResult:
    """Résultat d'extraction structuré"""
    extraction_type: str
    confidence: float
    extracted_data: Dict[str, Any]
    raw_matches: List[str]
    processing_time: float
    model_used: str


class SmolLM2ReasoningService:
    """Service de raisonnement basé sur les modèles Hugging Face"""
    
    def __init__(self, config):
        self.config = config
        self.model_name = config.REASONING_MODEL_NAME  # ← Model from config
        self.model = None
        self.tokenizer = None
        self.device = None
        self._is_loaded = False
        
        # Templates de prompts optimisés
        self.prompt_templates = self._init_prompt_templates()
        
        # Configuration d'optimisation
        self.generation_config = GenerationConfig(
            max_new_tokens=config.REASONING_MAX_TOKENS,
            temperature=config.REASONING_TEMPERATURE,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=None  # Sera défini après chargement du tokenizer
        )
    
    async def load_model(self, use_quantization: bool = True):
        """Charger le modèle de raisonnement avec optimisations"""
        try:
            logger.info(f"Chargement du modèle: {self.model_name}")
            
            # Device selection
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Utilisation device: {self.device}")
            
            # Configuration de quantization pour efficacité
            quantization_config = None
            if use_quantization and torch.cuda.is_available():
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_enable_fp32_cpu_offload=False
                )
                logger.info("Quantization 8-bit activée")
            
            # Charger le tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Ajouter token de padding si nécessaire
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Charger le modèle
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Configuration finale
            self.generation_config.pad_token_id = self.tokenizer.eos_token_id
            
            # Mode évaluation
            self.model.eval()
            
            self._is_loaded = True
            logger.info(f"Modèle {self.model_name} chargé avec succès")
            
            # Afficher les stats mémoire
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                cached = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"Mémoire GPU: {allocated:.2f}GB allouée, {cached:.2f}GB réservée")
                
        except Exception as e:
            logger.error(f"Erreur chargement {self.model_name}: {str(e)}")
            raise RuntimeError(f"Impossible de charger {self.model_name}: {str(e)}")
    
    def _init_prompt_templates(self) -> Dict[str, str]:
        """Initialiser le template de prompt custom optimisé"""
        return {
            ExtractionType.CUSTOM.value: """<|im_start|>system
                You are an intelligent data extraction assistant. Analyze the text according to the instructions.
                Respond ONLY with valid JSON.
                <|im_end|>
                <|im_start|>user
                Text to analyze:
                {text}
                
                Extraction instructions:
                {custom_instructions}

                RULES:
                    - Use null if information is not available
                    - Return ONLY valid JSON, no markdown formatting
                    - Do not include any additional text or explanations
                
                JSON Response:
                <|im_end|>
                <|im_start|>assistant"""
        }
    
    async def extract_information(
        self,
        text: str,
        extraction_type: Union[ExtractionType, str],
        custom_instructions: Optional[str] = None,
        max_length: int = 4096
    ) -> ExtractionResult:
        """
        Extraire des informations structurées du texte
        
        Args:
            text: Texte source (résultat OCR)
            extraction_type: Type d'extraction à effectuer
            custom_instructions: Instructions personnalisées pour extraction custom
            max_length: Longueur max du texte à traiter
            
        Returns:
            ExtractionResult avec données structurées
        """
        if not self._is_loaded:
            raise RuntimeError(f"Modèle {self.model_name} n'est pas chargé")
        
        import time
        start_time = time.time()
        
        try:
            # Préparer le type d'extraction
            if isinstance(extraction_type, str):
                extraction_type = ExtractionType(extraction_type)
            
            # Tronquer le texte si trop long
            if len(text) > max_length:
                text = text[:max_length] + "..."
                logger.warning(f"Texte tronqué à {max_length} caractères")
            
            # Construire le prompt
            prompt = self._build_prompt(text, extraction_type, custom_instructions)

            print(f"Prompt utilisé:\n{prompt}\n")
            
            # Tokenization
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            ).to(self.device)
            
            # Génération avec optimisations
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    use_cache=True,
                    do_sample=True
                )
            
            # Décoder la réponse
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()

            print(f"Réponse brute du modèle: \n======================== \n{response}\n========================\n")
            
            # Parser la réponse JSON
            extracted_data, confidence = self._parse_response(response)
            
            processing_time = time.time() - start_time
            
            # Créer le résultat
            result = ExtractionResult(
                extraction_type=extraction_type.value,
                confidence=confidence,
                extracted_data=extracted_data,
                raw_matches=self._find_raw_matches(text, extracted_data),
                processing_time=processing_time,
                model_used=self.model_name
            )
            
            logger.info(f"Extraction {extraction_type.value} terminée en {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Erreur extraction {self.model_name}: {str(e)}")
            
            # Retourner un résultat d'erreur
            return ExtractionResult(
                extraction_type=extraction_type.value if isinstance(extraction_type, ExtractionType) else extraction_type,
                confidence=0.0,
                extracted_data={"error": str(e)},
                raw_matches=[],
                processing_time=processing_time,
                model_used=self.model_name
            )
    
    def _build_prompt(
        self, 
        text: str, 
        extraction_type: ExtractionType,
        custom_instructions: Optional[str] = None
    ) -> str:
        """Construire le prompt optimisé pour extraction custom"""
        if extraction_type != ExtractionType.CUSTOM:
            raise ValueError("Only custom extraction type is supported")
            
        if not custom_instructions:
            raise ValueError("Custom instructions are required")
            
        template = self.prompt_templates[ExtractionType.CUSTOM.value]
        return template.format(text=text, custom_instructions=custom_instructions)
    
    def _parse_response(self, response: str) -> tuple[Dict[str, Any], float]:
        """Parser la réponse JSON du modèle"""
        try:
            # Nettoyer la réponse
            response = response.strip()
            
            # Chercher le JSON dans la réponse
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                try:
                    parsed_data = json.loads(json_str)
                    confidence = 1.0
                    return parsed_data, confidence
                except json.JSONDecodeError:
                    logger.warning(f"JSON invalide dans la réponse {self.model_name}")
            
            # Fallback: essayer de parser directement
            try:
                parsed_data = json.loads(response)
                confidence = self._calculate_confidence(parsed_data)
                return parsed_data, confidence
            except json.JSONDecodeError:
                # Dernier fallback: extraction basique
                logger.warning("Impossible de parser JSON, utilisation fallback")
                return {"raw_response": response}, 0.3
                
        except Exception as e:
            logger.error(f"Erreur parsing réponse: {str(e)}")
            return {"error": "Parsing failed", "raw_response": response}, 0.0
    
    
    def _find_raw_matches(self, text: str, extracted_data: Dict[str, Any]) -> List[str]:
        """Trouver les correspondances brutes dans le texte original"""
        matches = []
        
        def extract_values(obj):
            if isinstance(obj, dict):
                for value in obj.values():
                    extract_values(value)
            elif isinstance(obj, list):
                for item in obj:
                    extract_values(item)
            elif isinstance(obj, str) and len(obj) > 3:
                # Chercher la valeur dans le texte original
                if obj.lower() in text.lower():
                    matches.append(obj)
        
        extract_values(extracted_data)
        return list(set(matches))  # Enlever les doublons
    
    @property
    def is_loaded(self) -> bool:
        """Vérifier si le modèle est chargé"""
        return self._is_loaded and self.model is not None
    
    async def cleanup(self):
        """Nettoyer les ressources"""
        logger.info(f"Nettoyage des ressources {self.model_name}...")
        
        if self.model is not None:
            del self.model
            self.model = None
            
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self._is_loaded = False
        logger.info(f"Nettoyage {self.model_name} terminé")