# File: models/ocr_model.py
"""
OCR Model Manager for GOT-OCR 2.0
Handles model loading, initialization, and resource management
"""

import logging
from typing import Optional, List, Dict, Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

from config import Config

logger = logging.getLogger(__name__)


class OCRModelManager:
    """Manages OCR model loading and inference operations"""
    
    def __init__(self, config: Config):
        self.config = config
        self.model: Optional[AutoModelForImageTextToText] = None
        self.processor: Optional[AutoProcessor] = None
        self.device: Optional[torch.device] = None
        self._is_loaded = False
    
    async def load_model(self) -> None:
        """Load and initialize the OCR model"""
        try:
            logger.info(f"Loading OCR model: {self.config.MODEL_NAME}")
            
            # Determine device
            self.device = torch.device(self.config.device_preference)
            logger.info(f"Using device: {self.device}")
            
            # Load processor
            logger.info("Loading processor...")
            self.processor = AutoProcessor.from_pretrained(self.config.MODEL_NAME)
            
            # Load model
            logger.info("Loading model...")
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.config.MODEL_NAME,
                low_cpu_mem_usage=self.config.LOW_CPU_MEM_USAGE,
                device_map=str(self.device)
            )
            
            # Move model to device and set to evaluation mode
            self.model = self.model.eval().to(self.device)
            
            self._is_loaded = True
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")
    
    def generate_text(
        self,
        images: List[Image.Image],
        task_config: Dict[str, Any]
    ) -> str:
        """Generate text from images using the OCR model"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        try:
            # Prepare inputs based on task configuration
            inputs = self._prepare_inputs(images, task_config)
            
            # Generate text
            with torch.no_grad():
                generate_ids = self.model.generate(
                    **inputs,
                    do_sample=False,
                    tokenizer=self.processor.tokenizer,
                    stop_strings=self.config.STOP_STRINGS,
                    max_new_tokens=self.config.MAX_NEW_TOKENS,
                )
            
            # Decode the generated text
            result = self.processor.decode(
                generate_ids[0, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Text generation failed: {str(e)}")
            raise RuntimeError(f"Generation error: {str(e)}")
    
    def _prepare_inputs(self, images: List[Image.Image], task_config: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Prepare model inputs based on task configuration"""
        task = task_config.get("task")
        
        if task == "Plain Text OCR":
            inputs = self.processor(images, return_tensors="pt")
            
        elif task == "Format Text OCR":
            inputs = self.processor(images, return_tensors="pt", format=True)
            
        elif task == "Fine-grained OCR (Box)":
            box = task_config.get("box")
            if not box:
                raise ValueError("Box coordinates required for Fine-grained OCR (Box)")
            inputs = self.processor(images, return_tensors="pt", box=box)
            
        elif task == "Fine-grained OCR (Color)":
            color = task_config.get("color")
            if not color:
                raise ValueError("Color required for Fine-grained OCR (Color)")
            inputs = self.processor(images, return_tensors="pt", color=color)
            
        elif task == "Multi-page OCR":
            inputs = self.processor(
                images,
                return_tensors="pt",
                multi_page=True,
                format=True
            )
            
        else:
            raise ValueError(f"Unsupported task: {task}")
        
        # Move inputs to device
        return {k: v.to(self.device) for k, v in inputs.items()}
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._is_loaded and self.model is not None and self.processor is not None
    
    async def cleanup(self) -> None:
        """Cleanup model resources"""
        logger.info("Cleaning up model resources...")
        
        if self.model is not None:
            del self.model
            self.model = None
            
        if self.processor is not None:
            del self.processor
            self.processor = None
            
        # Clear CUDA cache if using GPU
        if self.device and self.device.type == "cuda":
            torch.cuda.empty_cache()
            
        self._is_loaded = False
        logger.info("Model cleanup completed")