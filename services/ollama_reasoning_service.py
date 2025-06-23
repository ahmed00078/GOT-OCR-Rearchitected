# File: services/ollama_reasoning_service.py
"""
Service de raisonnement avec Ollama (Gemma 2:12B)
Extraction d'informations structurées à partir du texte OCR
"""

import logging
import json
import time
import aiohttp
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

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


class OllamaReasoningService:
    """Service de raisonnement basé sur Ollama"""
    
    def __init__(self, config):
        self.config = config
        self.model_name = getattr(config, 'REASONING_MODEL_NAME', 'qwen3:8b')
        self.ollama_url = getattr(config, 'OLLAMA_URL', 'http://localhost:11434')
        self.max_tokens = getattr(config, 'REASONING_MAX_TOKENS', 4096)
        self.temperature = getattr(config, 'REASONING_TEMPERATURE', 0.1)
        self._is_loaded = False
        
    async def load_model(self, use_quantization: bool = True):
        """Charger et vérifier la disponibilité du modèle Ollama"""
        logger.info(f"Vérification du modèle Ollama: {self.model_name}")
        
        try:
            # Vérifier si Ollama est accessible
            async with aiohttp.ClientSession() as session:
                # Test de connexion
                async with session.get(f"{self.ollama_url}/api/version", timeout=10) as response:
                    if response.status != 200:
                        raise Exception(f"Ollama server non accessible: {response.status}")
                
                # Vérifier si le modèle est disponible
                async with session.get(f"{self.ollama_url}/api/tags") as response:
                    if response.status == 200:
                        models_data = await response.json()
                        available_models = [model['name'] for model in models_data.get('models', [])]
                        
                        if self.model_name not in available_models:
                            logger.warning(f"Modèle {self.model_name} non trouvé. Modèles disponibles: {available_models}")
                            # Tentative de pull du modèle
                            await self._pull_model()
                
                # Test de génération simple
                test_result = await self._test_generation()
                if test_result:
                    self._is_loaded = True
                    logger.info(f"Modèle Ollama {self.model_name} chargé avec succès")
                else:
                    raise Exception("Test de génération échoué")
                    
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle Ollama: {str(e)}")
            self._is_loaded = False
            raise
    
    async def _pull_model(self):
        """Télécharger le modèle si nécessaire"""
        logger.info(f"Téléchargement du modèle {self.model_name}...")
        
        async with aiohttp.ClientSession() as session:
            payload = {"name": self.model_name}
            async with session.post(
                f"{self.ollama_url}/api/pull",
                json=payload,
                timeout=300  # 5 minutes pour le téléchargement
            ) as response:
                if response.status != 200:
                    raise Exception(f"Échec du téléchargement du modèle: {response.status}")
                
                # Lire la réponse en streaming
                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line.decode())
                            if data.get('status'):
                                logger.info(f"Pull status: {data['status']}")
                        except:
                            pass
    
    async def _test_generation(self) -> bool:
        """Tester la génération avec une requête simple"""
        try:
            test_prompt = "Say 'test successful' if you can understand this."
            
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model_name,
                    "prompt": test_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 50
                    }
                }
                
                async with session.post(
                    f"{self.ollama_url}/api/generate",
                    json=payload,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        response_text = result.get('response', '').lower()
                        return 'test successful' in response_text or 'test' in response_text
                    
            return False
        except Exception as e:
            logger.error(f"Test de génération échoué: {str(e)}")
            return False
    
    async def extract_information(
        self,
        text: str,
        extraction_type: ExtractionType,
        custom_instructions: Optional[str] = None,
        max_length: int = 4096
    ) -> ExtractionResult:
        """Extraire des informations du texte avec Ollama"""
        
        if not self._is_loaded:
            raise RuntimeError("Le modèle Ollama n'est pas chargé")
        
        start_time = time.time()
        
        try:
            # Construire le prompt
            prompt = self._build_extraction_prompt(text, extraction_type, custom_instructions, max_length)
            
            # Générer la réponse avec Ollama
            response_text = await self._generate_with_ollama(prompt)
            
            # Traiter la réponse
            extracted_data = self._parse_ollama_response(response_text)
            
            processing_time = time.time() - start_time
            
            # Calculer la confiance basée sur la qualité de la réponse
            confidence = self._calculate_confidence(extracted_data, response_text)
            
            return ExtractionResult(
                extraction_type=extraction_type.value,
                confidence=confidence,
                extracted_data=extracted_data,
                raw_matches=[response_text],
                processing_time=processing_time,
                model_used=self.model_name
            )
            
        except Exception as e:
            logger.error(f"Erreur extraction Ollama: {str(e)}")
            processing_time = time.time() - start_time
            
            return ExtractionResult(
                extraction_type=extraction_type.value,
                confidence=0.0,
                extracted_data={"error": str(e)},
                raw_matches=[],
                processing_time=processing_time,
                model_used=self.model_name
            )
    
    def _build_extraction_prompt(
        self,
        text: str,
        extraction_type: ExtractionType,
        custom_instructions: Optional[str],
        max_length: int
    ) -> str:
        """Construire le prompt d'extraction optimisé pour Ollama"""
        
        # Truncate text si trop long
        if len(text) > max_length:
            text = text[:max_length] + "..."
        
        if extraction_type == ExtractionType.CUSTOM and custom_instructions:
            instructions = custom_instructions
        else:
            instructions = "Extract all relevant information from the document"
        
        prompt = f"""You are an expert data extraction AI. Your task is to analyze the following document text and extract structured information.

INSTRUCTIONS: {instructions}

DOCUMENT TEXT:
{text}

Please extract the requested information and format your response as a JSON object. Be precise and only include information that is clearly present in the text. If certain information is not available, mark it as null.

Response format:
{{
    "extracted_information": {{
        // Your extracted data here
    }},
    "summary": "Brief summary of what was found",
    "confidence_notes": "Explanation of extraction confidence"
}}

JSON Response:"""
        
        return prompt
    
    async def _generate_with_ollama(self, prompt: str) -> str:
        """Générer une réponse avec l'API Ollama"""
        
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            }
            
            async with session.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=120  # 2 minutes timeout
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error {response.status}: {error_text}")
                
                result = await response.json()
                return result.get('response', '')
    
    def _parse_ollama_response(self, response_text: str) -> Dict[str, Any]:
        """Parser la réponse JSON de Ollama"""
        
        try:
            # Chercher le JSON dans la réponse
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end != 0:
                json_text = response_text[json_start:json_end]
                parsed = json.loads(json_text)
                
                # Extraire les données principales
                if 'extracted_information' in parsed:
                    return parsed['extracted_information']
                else:
                    return parsed
            else:
                # Fallback: retourner le texte brut
                return {
                    "raw_text": response_text,
                    "note": "Could not parse as JSON"
                }
                
        except json.JSONDecodeError:
            # Fallback: analyser le texte manuellement
            logger.warning("Impossible de parser la réponse JSON, analyse manuelle...")
            return self._manual_text_analysis(response_text)
        except Exception as e:
            logger.error(f"Erreur parsing réponse: {str(e)}")
            return {"error": str(e), "raw_response": response_text}
    
    def _manual_text_analysis(self, text: str) -> Dict[str, Any]:
        """Analyse manuelle du texte si le JSON parsing échoue"""
        
        extracted = {}
        
        # Extraire des patterns communs
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        price_pattern = r'\$\d+(?:\.\d{2})?|\d+(?:\.\d{2})?\s*(?:EUR|USD|€|\$)'
        
        emails = re.findall(email_pattern, text, re.IGNORECASE)
        phones = re.findall(phone_pattern, text)
        prices = re.findall(price_pattern, text, re.IGNORECASE)
        
        if emails:
            extracted['emails'] = emails
        if phones:
            extracted['phones'] = phones
        if prices:
            extracted['prices'] = prices
        
        # Ajouter le texte brut pour référence
        extracted['raw_analysis'] = text[:500]  # Premier 500 caractères
        
        return extracted
    
    def _calculate_confidence(self, extracted_data: Dict[str, Any], response_text: str) -> float:
        """Calculer un score de confiance basé sur la qualité de l'extraction"""
        
        confidence = 0.5  # Base confidence
        
        # Bonus si JSON valide
        if 'error' not in extracted_data and 'raw_text' not in extracted_data:
            confidence += 0.2
        
        # Bonus si des données ont été extraites
        if extracted_data and len(extracted_data) > 0:
            confidence += 0.2
        
        # Bonus si la réponse est structurée
        if any(key in response_text.lower() for key in ['json', '{', '}', '"']):
            confidence += 0.1
        
        # Malus si erreurs
        if 'error' in extracted_data:
            confidence -= 0.3
        
        return max(0.0, min(1.0, confidence))
    
    async def cleanup(self):
        """Nettoyer les ressources"""
        logger.info("Nettoyage du service de raisonnement Ollama")
        self._is_loaded = False
    
    @property
    def is_loaded(self) -> bool:
        """Vérifier si le modèle est chargé"""
        return self._is_loaded