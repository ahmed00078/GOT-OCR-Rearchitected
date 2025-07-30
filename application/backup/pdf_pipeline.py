#!/usr/bin/env python3
"""
PDF Processing Pipeline
Clean, efficient pipeline for batch PDF processing with custom information extraction
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import sys
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from models.ocr_model import OCRModelManager
from services.enhanced_ocr_service import EnhancedOCRService
from fastapi import UploadFile, BackgroundTasks
import io


class PDFPipeline:
    """Clean PDF processing pipeline with custom extraction"""
    
    def __init__(self):
        self.config = Config()
        self.model_manager = None
        self.enhanced_service = None
        self.results = []
        
    async def initialize(self):
        """Initialize OCR and AI models"""
        print("🔄 Initializing models...")
        
        self.model_manager = OCRModelManager(self.config)
        await self.model_manager.load_model()
        
        self.enhanced_service = EnhancedOCRService(
            model_manager=self.model_manager,
            config=self.config
        )
        
        if self.config.reasoning_enabled:
            await self.enhanced_service.initialize_reasoning()
        
        print("✅ Models initialized successfully")
    
    async def process_single_pdf(self, pdf_path: Path, custom_prompt: str) -> Dict[str, Any]:
        """Process a single PDF with custom extraction prompt"""
        try:
            print(f"📄 Processing: {pdf_path.name}")
            
            # Convert PDF to UploadFile
            with open(pdf_path, 'rb') as f:
                content = f.read()
                upload_file = UploadFile(
                    filename=pdf_path.name,
                    file=io.BytesIO(content)
                )
            
            start_time = time.time()
            
            # Process with AI reasoning
            result = await self.enhanced_service.process_with_reasoning(
                task="Multi-page OCR",
                images=[upload_file],
                background_tasks=BackgroundTasks(),
                extraction_type="custom",
                custom_instructions=custom_prompt
            )
            
            processing_time = time.time() - start_time
            
            return {
                "filename": pdf_path.name,
                "success": True,
                "processing_time": round(processing_time, 2),
                "extracted_text": result.get("text", ""),
                "extracted_data": result.get("extraction_result", {}).get("extracted_data", {}),
                "text_length": len(result.get("text", "")),
                "error": None
            }
            
        except Exception as e:
            return {
                "filename": pdf_path.name,
                "success": False,
                "processing_time": 0,
                "extracted_text": "",
                "extracted_data": {},
                "text_length": 0,
                "error": str(e)
            }
    
    async def run_batch_processing(self, pdf_folder: str, custom_prompt: str, output_file: str = None) -> List[Dict[str, Any]]:
        """Run batch processing on all PDFs in folder"""
        
        pdf_path = Path(pdf_folder)
        if not pdf_path.exists():
            raise ValueError(f"PDF folder not found: {pdf_folder}")
        
        pdf_files = list(pdf_path.glob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in: {pdf_folder}")
        
        print(f"📁 Found {len(pdf_files)} PDF files")
        print(f"🎯 Custom extraction prompt: {custom_prompt[:100]}...")
        
        results = []
        total_start = time.time()
        
        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"\n[{i}/{len(pdf_files)}] Processing {pdf_file.name}")
            
            result = await self.process_single_pdf(pdf_file, custom_prompt)
            results.append(result)
            
            # Show progress
            if result["success"]:
                data = result["extracted_data"]
                fields_found = len([k for k, v in data.items() if v and str(v).lower() not in ['null', 'none', '']])
                print(f"  ✅ Success in {result['processing_time']}s")
                print(f"  📊 Extracted {fields_found} fields from {result['text_length']} characters")
            else:
                print(f"  ❌ Failed: {result['error']}")
        
        total_time = time.time() - total_start
        
        # Summary statistics
        successful = [r for r in results if r["success"]]
        print(f"\n📈 PIPELINE RESULTS")
        print(f"⏱️  Total time: {total_time:.2f}s")
        print(f"✅ Successful: {len(successful)}/{len(results)}")
        print(f"⚡ Average time: {total_time/len(results):.2f}s per file")
        
        # Save results
        if output_file:
            output_path = Path(output_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"💾 Results saved to: {output_path}")
        
        self.results = results
        return results
    
    def export_to_csv(self, output_file: str):
        """Export results to CSV format"""
        import pandas as pd
        
        if not self.results:
            raise ValueError("No results to export. Run processing first.")
        
        # Flatten results for CSV
        csv_data = []
        for result in self.results:
            row = {
                "filename": result["filename"],
                "success": result["success"],
                "processing_time": result["processing_time"],
                "text_length": result["text_length"],
                "error": result["error"]
            }
            
            # Add extracted fields
            if result["success"] and result["extracted_data"]:
                for key, value in result["extracted_data"].items():
                    row[f"extracted_{key}"] = value
            
            csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        df.to_csv(output_file, index=False)
        print(f"📊 CSV exported to: {output_file}")
    
    def get_extraction_summary(self) -> Dict[str, Any]:
        """Get summary of extraction results"""
        if not self.results:
            return {}
        
        successful = [r for r in self.results if r["success"]]
        total_files = len(self.results)
        
        # Analyze extracted fields
        all_fields = set()
        field_counts = {}
        
        for result in successful:
            data = result.get("extracted_data", {})
            for field, value in data.items():
                all_fields.add(field)
                if value and str(value).lower() not in ['null', 'none', '']:
                    field_counts[field] = field_counts.get(field, 0) + 1
        
        return {
            "total_files": total_files,
            "successful_files": len(successful),
            "success_rate": len(successful) / total_files * 100,
            "total_processing_time": sum(r["processing_time"] for r in successful),
            "average_processing_time": sum(r["processing_time"] for r in successful) / len(successful) if successful else 0,
            "extracted_fields": dict(field_counts),
            "field_extraction_rates": {
                field: (count / total_files * 100) 
                for field, count in field_counts.items()
            }
        }


def main():
    """Main entry point with example usage"""
    print("🔧 PDF Processing Pipeline")
    print("=" * 50)
    
    # Example usage
    pipeline = PDFPipeline()
    
    # Example custom prompts for different use cases
    EXAMPLE_PROMPTS = {
        "equipment_specs": """
Extract electronic equipment information:
- manufacturer: Company name (Apple, Microsoft, Lenovo, etc.)
- year: Product year or document year
- product_name: Product model/name
- carbon_footprint: CO2 emissions in kg
- power_consumption: Electrical consumption in Watts
- weight: Product weight in kg
Return JSON with these exact field names. Use null for missing data.
""",
        
        "financial_data": """
Extract financial information:
- company: Company name
- revenue: Annual revenue with currency
- costs: Operating costs
- profit: Net profit/loss
- currency: Main currency used
- year: Financial year
Return JSON format only.
""",
        
        "contact_info": """
Extract contact information:
- company: Organization name
- email: Email addresses
- phone: Phone numbers
- address: Physical addresses
- website: Web URLs
- contact_person: Key contacts
Return JSON format only.
"""
    }
    
    print("Available extraction templates:")
    for key, prompt in EXAMPLE_PROMPTS.items():
        print(f"  - {key}: {prompt.split(':')[1].split('Return')[0].strip()}")
    
    print("\nTo use this pipeline:")
    print("1. Create PDFPipeline instance")
    print("2. Call initialize() to load models")
    print("3. Call run_batch_processing() with folder and prompt")
    print("4. Use export_to_csv() or get_extraction_summary() for results")


if __name__ == "__main__":
    main()