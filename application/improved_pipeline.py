#!/usr/bin/env python3
"""
Improved PDF Processing Pipeline
Enhanced with better JSON validation, retry logic, and document-specific prompts
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys
import os
import re

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from models.ocr_model import OCRModelManager
from services.enhanced_ocr_service import EnhancedOCRService
from fastapi import UploadFile, BackgroundTasks
import io


class DocumentTypeDetector:
    """Detect document type to use appropriate extraction strategy"""
    
    @staticmethod
    def detect_type(text: str, filename: str) -> str:
        """Detect document type based on content and filename"""
        text_lower = text.lower()
        filename_lower = filename.lower()
        
        # Apple documents
        if any(keyword in text_lower for keyword in ['ecoprofile', 'product environmental report']) or \
           any(keyword in filename_lower for keyword in ['macbook', 'imac', 'apple']):
            return 'apple'
        
        # HP documents  
        elif any(keyword in text_lower for keyword in ['product carbon footprint', 'paia tool']) or \
             any(keyword in filename_lower for keyword in ['hp', 'c0']):
            return 'hp'
        
        # Lenovo documents
        elif any(keyword in text_lower for keyword in ['pcf-', 'lenovo', 'ideapad', 'thinkpad']) or \
             filename_lower.startswith('pcf-'):
            return 'lenovo'
        
        # Acer documents
        elif any(keyword in text_lower for keyword in ['acer', 'veriton']) or \
             any(keyword in filename_lower for keyword in ['acer', 'veriton']):
            return 'acer'
        
        # Microsoft documents
        elif any(keyword in text_lower for keyword in ['microsoft', 'surface']) or \
             any(keyword in filename_lower for keyword in ['surface', 'xbox']):
            return 'microsoft'
        
        return 'generic'


class ImprovedPromptGenerator:
    """Generate optimized prompts based on document type"""
    
    BASE_FIELDS = {
        "manufacturer": "Company name (Apple, Microsoft, Lenovo, Acer, HP, etc.)",
        "year": "Product year or document year (2020, 2021, 2022, 2023, 2024)",
        "product_name": "Product model or name",
        "carbon_footprint": "CO2 emissions or carbon footprint (kg CO2 or CO2eq)",
        "power_consumption": "Electrical consumption (Watts, W, kWh)",
        "weight": "Product weight (kg, grams)"
    }
    
    @classmethod
    def generate_strict_prompt(cls, doc_type: str = 'generic') -> str:
        """Generate strict JSON-constrained prompt"""
        
        # Document-specific hints
        doc_hints = {
            'apple': "Look for: MacBook, iMac, Product Environmental Report, carbon footprint in kg CO2e",
            'hp': "Look for: HP brand, Product Carbon Footprint, PAIA tool results, TEC values",
            'lenovo': "Look for: Lenovo, IdeaPad, ThinkPad, PCF values, carbon footprint estimates", 
            'acer': "Look for: Acer brand, Product Carbon Footprint, PAIA algorithm, typical energy consumption",
            'microsoft': "Look for: Microsoft, Surface, Xbox, carbon emissions, power consumption modes",
            'generic': "Look for manufacturer names, product models, carbon footprint values, power specs"
        }
        
        hint = doc_hints.get(doc_type, doc_hints['generic'])
        
        return f"""CRITICAL INSTRUCTIONS - FOLLOW EXACTLY:

{hint}

Extract ONLY these 6 fields in this EXACT JSON format:

{{
  "manufacturer": "string or null",
  "year": "string or null",
  "product_name": "string or null", 
  "carbon_footprint": "number or null",
  "power_consumption": "number or null",
  "weight": "number or null"
}}

FIELD DEFINITIONS:
- manufacturer: {cls.BASE_FIELDS['manufacturer']}
- year: {cls.BASE_FIELDS['year']}
- product_name: {cls.BASE_FIELDS['product_name']}
- carbon_footprint: {cls.BASE_FIELDS['carbon_footprint']} (extract only numeric value)
- power_consumption: {cls.BASE_FIELDS['power_consumption']} (extract only numeric value)
- weight: {cls.BASE_FIELDS['weight']} (extract only numeric value)

CRITICAL RULES:
1. Response must be VALID JSON only - no markdown, no explanations
2. Use EXACTLY these 6 field names - no additional fields allowed
3. Use null for missing information
4. Extract only numeric values for carbon_footprint, power_consumption, weight
5. Do not include units in numeric fields
6. Do not add any text before or after the JSON

EXAMPLE VALID RESPONSE:
{{"manufacturer": "Apple", "year": "2023", "product_name": "MacBook Air M2", "carbon_footprint": 161, "power_consumption": 67, "weight": 1.24}}

JSON Response:"""


class JSONValidator:
    """Validate and clean JSON responses"""
    
    ALLOWED_FIELDS = {'manufacturer', 'year', 'product_name', 'carbon_footprint', 'power_consumption', 'weight'}
    
    @classmethod
    def validate_and_clean(cls, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and clean extracted data"""
        
        # If raw_response exists, it means JSON parsing failed
        if 'raw_response' in response_data:
            # Try to extract JSON from raw response
            cleaned = cls._extract_json_from_text(response_data['raw_response'])
            if cleaned:
                return cls._filter_and_validate_fields(cleaned)
            else:
                return cls._create_empty_response()
        
        return cls._filter_and_validate_fields(response_data)
    
    @classmethod
    def _extract_json_from_text(cls, text: str) -> Optional[Dict[str, Any]]:
        """Try to extract JSON from malformed text response"""
        try:
            # Find JSON-like patterns
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, text, re.DOTALL)
            
            for match in matches:
                try:
                    parsed = json.loads(match)
                    if isinstance(parsed, dict):
                        return parsed
                except:
                    continue
            
            # Try to parse the entire text as JSON
            return json.loads(text)
            
        except:
            return None
    
    @classmethod
    def _filter_and_validate_fields(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter to allowed fields and validate types"""
        cleaned = {}
        
        for field in cls.ALLOWED_FIELDS:
            value = data.get(field)
            
            if value is None or str(value).lower() in ['null', 'none', '']:
                cleaned[field] = None
            else:
                # Type validation and conversion
                if field in ['carbon_footprint', 'power_consumption', 'weight']:
                    cleaned[field] = cls._extract_numeric_value(str(value))
                else:
                    cleaned[field] = str(value).strip()
        
        return cleaned
    
    @classmethod
    def _extract_numeric_value(cls, text: str) -> Optional[float]:
        """Extract numeric value from text with units"""
        try:
            # Remove common units and extract number
            text = re.sub(r'[^\d.,]', '', text)
            text = text.replace(',', '.')
            
            # Find first number
            number_match = re.search(r'\d+\.?\d*', text)
            if number_match:
                return float(number_match.group())
        except:
            pass
        return None
    
    @classmethod
    def _create_empty_response(cls) -> Dict[str, Any]:
        """Create response with all null values"""
        return {field: None for field in cls.ALLOWED_FIELDS}


class ImprovedPDFPipeline:
    """Enhanced PDF processing pipeline with better error handling"""
    
    def __init__(self):
        self.config = Config()
        self.model_manager = None
        self.enhanced_service = None
        self.results = []
        self.prompt_generator = ImprovedPromptGenerator()
        self.detector = DocumentTypeDetector()
        self.validator = JSONValidator()
        
    async def initialize(self):
        """Initialize OCR and AI models"""
        print("🔄 Initializing improved pipeline...")
        
        self.model_manager = OCRModelManager(self.config)
        await self.model_manager.load_model()
        
        self.enhanced_service = EnhancedOCRService(
            model_manager=self.model_manager,
            config=self.config
        )
        
        if self.config.reasoning_enabled:
            await self.enhanced_service.initialize_reasoning()
        
        print("✅ Enhanced pipeline initialized successfully")
    
    async def process_single_pdf_with_retry(self, pdf_path: Path) -> Dict[str, Any]:
        """Process PDF with retry logic and validation"""
        
        print(f"📄 Processing: {pdf_path.name}")
        
        # First attempt
        result = await self._attempt_processing(pdf_path, attempt=1)
        
        # Check if retry is needed
        if self._needs_retry(result):
            print(f"  🔄 Retrying with stricter prompt...")
            result = await self._attempt_processing(pdf_path, attempt=2, strict=True)
        
        return result
    
    async def _attempt_processing(self, pdf_path: Path, attempt: int, strict: bool = False) -> Dict[str, Any]:
        """Single processing attempt"""
        try:
            # Convert PDF to UploadFile
            with open(pdf_path, 'rb') as f:
                content = f.read()
                upload_file = UploadFile(
                    filename=pdf_path.name,
                    file=io.BytesIO(content)
                )
            
            start_time = time.time()
            
            # Get text first for document type detection
            ocr_result = await self.enhanced_service._process_with_layout(
                task="Multi-page OCR",
                images=[upload_file],
                background_tasks=BackgroundTasks()
            )
            
            extracted_text = ocr_result.get("text", "")
            doc_type = self.detector.detect_type(extracted_text, pdf_path.name)
            
            # Generate appropriate prompt
            if strict:
                custom_prompt = self.prompt_generator.generate_strict_prompt('generic')
            else:
                custom_prompt = self.prompt_generator.generate_strict_prompt(doc_type)
            
            # Reset upload file
            upload_file.file.seek(0)
            
            # Process with AI reasoning
            result = await self.enhanced_service.process_with_reasoning(
                task="Multi-page OCR",
                images=[upload_file],
                background_tasks=BackgroundTasks(),
                extraction_type="custom",
                custom_instructions=custom_prompt
            )
            
            processing_time = time.time() - start_time
            
            # Validate and clean the extraction
            raw_extracted = result.get("extraction_result", {}).get("extracted_data", {})
            cleaned_data = self.validator.validate_and_clean(raw_extracted)
            
            return {
                "filename": pdf_path.name,
                "success": True,
                "processing_time": round(processing_time, 2),
                "extracted_text": extracted_text,
                "extracted_data": cleaned_data,
                "text_length": len(extracted_text),
                "document_type": doc_type,
                "attempt": attempt,
                "error": None
            }
            
        except Exception as e:
            return {
                "filename": pdf_path.name,
                "success": False,
                "processing_time": 0,
                "extracted_text": "",
                "extracted_data": self.validator._create_empty_response(),
                "text_length": 0,
                "document_type": "unknown",
                "attempt": attempt,
                "error": str(e)
            }
    
    def _needs_retry(self, result: Dict[str, Any]) -> bool:
        """Check if processing needs retry"""
        if not result["success"]:
            return True
        
        extracted = result["extracted_data"]
        
        # Check if we got any valid data
        valid_fields = sum(1 for v in extracted.values() if v is not None)
        
        # Retry if we got 0 or 1 fields only
        return valid_fields <= 1
    
    async def run_batch_processing(self, pdf_folder: str, output_file: str = None) -> List[Dict[str, Any]]:
        """Run improved batch processing"""
        
        pdf_path = Path(pdf_folder)
        if not pdf_path.exists():
            raise ValueError(f"PDF folder not found: {pdf_folder}")
        
        pdf_files = list(pdf_path.glob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in: {pdf_folder}")
        
        print(f"📁 Found {len(pdf_files)} PDF files")
        print("🔧 Using improved pipeline with:")
        print("  - Document type detection")
        print("  - Strict JSON validation") 
        print("  - Retry logic for failed extractions")
        print("  - Field filtering and cleaning")
        
        results = []
        total_start = time.time()
        retries_count = 0
        
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"\n[{i}/{len(pdf_files)}]", end=" ")
            
            result = await self.process_single_pdf_with_retry(pdf_file)
            results.append(result)
            
            if result["attempt"] > 1:
                retries_count += 1
            
            # Show progress
            if result["success"]:
                data = result["extracted_data"]
                fields_found = len([k for k, v in data.items() if v is not None])
                print(f"  ✅ Success in {result['processing_time']}s ({result['document_type']})")
                print(f"  📊 Extracted {fields_found}/6 fields")
                
                # Show extracted fields
                for field, value in data.items():
                    if value is not None:
                        print(f"    - {field}: {value}")
            else:
                print(f"  ❌ Failed: {result['error']}")
        
        total_time = time.time() - total_start
        
        # Enhanced statistics
        successful = [r for r in results if r["success"]]
        print(f"\n📈 IMPROVED PIPELINE RESULTS")
        print(f"⏱️  Total time: {total_time:.2f}s")
        print(f"✅ Successful: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")
        print(f"🔄 Retries needed: {retries_count}")
        print(f"⚡ Average time: {total_time/len(results):.2f}s per file")
        
        # Document type breakdown
        doc_types = {}
        for result in results:
            doc_type = result.get("document_type", "unknown")
            if doc_type not in doc_types:
                doc_types[doc_type] = {"total": 0, "successful": 0}
            doc_types[doc_type]["total"] += 1
            if result["success"]:
                doc_types[doc_type]["successful"] += 1
        
        print(f"\n📋 BY DOCUMENT TYPE:")
        for doc_type, stats in doc_types.items():
            success_rate = stats["successful"] / stats["total"] * 100
            print(f"  {doc_type}: {stats['successful']}/{stats['total']} ({success_rate:.1f}%)")
        
        # Field extraction statistics
        field_stats = {}
        for field in self.validator.ALLOWED_FIELDS:
            field_stats[field] = sum(1 for r in successful 
                                   if r["extracted_data"].get(field) is not None)
        
        print(f"\n🎯 FIELD EXTRACTION RATES:")
        for field, count in field_stats.items():
            percentage = (count / len(results)) * 100
            status = "🟢" if percentage >= 70 else "🟡" if percentage >= 40 else "🔴"
            print(f"  {status} {field}: {count}/{len(results)} ({percentage:.1f}%)")
        
        # Save results
        if output_file:
            output_path = Path(output_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Results saved to: {output_path}")
        
        self.results = results
        return results
    
    def get_improvement_stats(self, old_results_file: str) -> Dict[str, Any]:
        """Compare with previous results"""
        try:
            with open(old_results_file, 'r') as f:
                old_results = json.load(f)
            
            # Calculate improvements
            old_successful = len([r for r in old_results if r.get("success", False)])
            new_successful = len([r for r in self.results if r["success"]])
            
            old_field_counts = {}
            new_field_counts = {}
            
            for field in self.validator.ALLOWED_FIELDS:
                old_field_counts[field] = 0
                new_field_counts[field] = 0
                
                for r in old_results:
                    if r.get("success") and r.get("extracted_data", {}).get(field):
                        old_field_counts[field] += 1
                
                for r in self.results:
                    if r["success"] and r["extracted_data"].get(field) is not None:
                        new_field_counts[field] += 1
            
            return {
                "old_success_rate": old_successful / len(old_results) * 100,
                "new_success_rate": new_successful / len(self.results) * 100,
                "improvement": (new_successful - old_successful) / len(old_results) * 100,
                "field_improvements": {
                    field: new_field_counts[field] - old_field_counts[field]
                    for field in self.validator.ALLOWED_FIELDS
                }
            }
        except:
            return {}


if __name__ == "__main__":
    print("🔧 Improved PDF Processing Pipeline")
    print("Enhanced with validation, retry logic, and document-specific prompts")