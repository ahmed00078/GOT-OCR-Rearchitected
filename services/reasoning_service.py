# File: services/reasoning_service.py
"""
Service de raisonnement avec SmolLM2:1.7B
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
    CARBON_FOOTPRINT = "carbon_footprint"  # Pour les données carbone
    TECHNICAL_SPECS = "technical_specs"    # Spécifications techniques
    FINANCIAL_DATA = "financial_data"      # Données financières
    CONTACT_INFO = "contact_info"          # Informations de contact
    CUSTOM = "custom"                      # Extraction personnalisée


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
    """Service de raisonnement basé sur SmolLM2:1.7B optimisé"""
    
    def __init__(self, config):
        self.config = config
        self.model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
        self.model = None
        self.tokenizer = None
        self.device = None
        self._is_loaded = False
        
        # Templates de prompts optimisés
        self.prompt_templates = self._init_prompt_templates()
        
        # Configuration d'optimisation
        self.generation_config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.1,  # Très déterministe pour l'extraction
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=None  # Sera défini après chargement du tokenizer
        )
    
    async def load_model(self, use_quantization: bool = True):
        """Charger SmolLM2 avec optimisations"""
        try:
            logger.info(f"Chargement de SmolLM2: {self.model_name}")
            
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
            logger.info("SmolLM2 chargé avec succès")
            
            # Afficher les stats mémoire
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                cached = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"Mémoire GPU: {allocated:.2f}GB allouée, {cached:.2f}GB réservée")
                
        except Exception as e:
            logger.error(f"Erreur chargement SmolLM2: {str(e)}")
            raise RuntimeError(f"Impossible de charger SmolLM2: {str(e)}")
    
    def _init_prompt_templates(self) -> Dict[str, str]:
        """Initialiser les templates de prompts optimisés"""
        return {
            ExtractionType.CARBON_FOOTPRINT.value: """<|im_start|>system
Vous êtes un expert en analyse de données environnementales. Extrayez les informations de bilan carbone à partir du texte fourni.
Répondez UNIQUEMENT avec un JSON valide contenant les champs trouvés.
<|im_end|>
<|im_start|>user
Texte à analyser:
{text}

Extrayez les informations de bilan carbone (émissions CO2, consommation énergétique, etc.) sous format JSON:
{{
  "carbon_emissions": "valeur avec unité",
  "energy_consumption": "valeur avec unité", 
  "product_name": "nom du produit",
  "manufacturer": "fabricant",
  "certification": "certifications environnementales",
  "additional_metrics": {{}}
}}
<|im_end|>
<|im_start|>assistant""",

            ExtractionType.TECHNICAL_SPECS.value: """<|im_start|>system
Vous êtes un expert en spécifications techniques. Extrayez les caractéristiques techniques à partir du texte.
Répondez UNIQUEMENT avec un JSON valide.
<|im_end|>
<|im_start|>user
Texte à analyser:
{text}

Extrayez les spécifications techniques sous format JSON:
{{
  "product_name": "nom",
  "model": "modèle",
  "dimensions": "dimensions",
  "weight": "poids",
  "power": "consommation électrique",
  "performance": {{}},
  "connectivity": [],
  "additional_specs": {{}}
}}
<|im_end|>
<|im_start|>assistant""",

            ExtractionType.FINANCIAL_DATA.value: """<|im_start|>system
Vous êtes un expert en analyse financière. Extrayez les données financières du texte.
Répondez UNIQUEMENT avec un JSON valide.
<|im_end|>
<|im_start|>user
Texte à analyser:
{text}

Extrayez les données financières sous format JSON:
{{
  "prices": [],
  "currencies": [],
  "financial_metrics": {{}},
  "costs": {{}},
  "revenue": "revenus",
  "expenses": "dépenses",
  "additional_financial_info": {{}}
}}
<|im_end|>
<|im_start|>assistant""",

            ExtractionType.CONTACT_INFO.value: """<|im_start|>system
Vous êtes un expert en extraction d'informations de contact. Trouvez tous les détails de contact.
Répondez UNIQUEMENT avec un JSON valide.
<|im_end|>
<|im_start|>user
Texte à analyser:
{text}

Extrayez les informations de contact sous format JSON:
{{
  "names": [],
  "emails": [],
  "phones": [],
  "addresses": [],
  "companies": [],
  "websites": [],
  "social_media": {{}}
}}
<|im_end|>
<|im_start|>assistant""",

            ExtractionType.CUSTOM.value: """<|im_start|>system
Vous êtes un assistant intelligent d'extraction de données. Analysez le texte selon les instructions.
Répondez UNIQUEMENT avec un JSON valide.
<|im_end|>
<|im_start|>user
Texte à analyser:
{text}

Instructions d'extraction:
{custom_instructions}

Répondez avec un JSON structuré contenant les informations demandées.
<|im_end|>
<|im_start|>assistant"""
        }
    
    async def extract_information(
        self,
        text: str,
        extraction_type: Union[ExtractionType, str],
        custom_instructions: Optional[str] = None,
        max_length: int = 2000
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
            raise RuntimeError("SmolLM2 n'est pas chargé")
        
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
            logger.error(f"Erreur extraction SmolLM2: {str(e)}")
            
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
        """Construire le prompt optimisé"""
        template = self.prompt_templates[extraction_type.value]
        
        if extraction_type == ExtractionType.CUSTOM and custom_instructions:
            return template.format(text=text, custom_instructions=custom_instructions)
        else:
            return template.format(text=text)
    
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
                    confidence = self._calculate_confidence(parsed_data)
                    return parsed_data, confidence
                except json.JSONDecodeError:
                    logger.warning("JSON invalide dans la réponse SmolLM2")
            
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
    
    def _calculate_confidence(self, data: Dict[str, Any]) -> float:
        """Calculer un score de confiance basé sur la complétude des données"""
        if not data or "error" in data:
            return 0.0
        
        # Compter les champs non vides
        filled_fields = 0
        total_fields = 0
        
        for key, value in data.items():
            total_fields += 1
            if value and value != "" and value != [] and value != {}:
                filled_fields += 1
        
        base_confidence = filled_fields / max(total_fields, 1) if total_fields > 0 else 0
        
        # Bonus pour les champs spécifiques importants
        important_fields = ["product_name", "model", "carbon_emissions", "price"]
        bonus = sum(0.1 for field in important_fields if field in data and data[field])
        
        return min(base_confidence + bonus, 1.0)
    
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
    
    async def batch_extract(
        self,
        texts: List[str],
        extraction_type: Union[ExtractionType, str],
        custom_instructions: Optional[str] = None
    ) -> List[ExtractionResult]:
        """Traitement par batch pour efficacité"""
        results = []
        
        # Traiter par petits batches pour éviter la surcharge mémoire
        batch_size = 3  # Petit batch pour SmolLM2:1.7B
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_results = []
            
            for text in batch:
                result = await self.extract_information(
                    text, extraction_type, custom_instructions
                )
                batch_results.append(result)
            
            results.extend(batch_results)
            
            # Petit délai pour éviter la surchauffe
            if len(texts) > batch_size:
                await asyncio.sleep(0.1)
        
        return results
    
    @property
    def is_loaded(self) -> bool:
        """Vérifier si le modèle est chargé"""
        return self._is_loaded and self.model is not None
    
    async def cleanup(self):
        """Nettoyer les ressources"""
        logger.info("Nettoyage des ressources SmolLM2...")
        
        if self.model is not None:
            del self.model
            self.model = None
            
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self._is_loaded = False
        logger.info("Nettoyage SmolLM2 terminé")


# === FONCTIONS UTILITAIRES ===

def create_reasoning_service(config) -> SmolLM2ReasoningService:
    """Factory function pour créer le service de raisonnement"""
    return SmolLM2ReasoningService(config)

async def quick_extract_carbon_data(reasoning_service, ocr_text: str) -> Dict[str, Any]:
    """Fonction de convenance pour extraction rapide de données carbone"""
    if not reasoning_service.is_loaded:
        await reasoning_service.load_model()
    
    result = await reasoning_service.extract_information(
        ocr_text, 
        ExtractionType.CARBON_FOOTPRINT
    )
    
    return {
        "carbon_data": result.extracted_data,
        "confidence": result.confidence,
        "processing_time": result.processing_time
    }