#!/usr/bin/env python3
"""
Pipeline Runner - Simple interface to execute PDF processing
Usage examples for the PDF pipeline
"""

import asyncio
import argparse
from pathlib import Path
from pdf_pipeline import PDFPipeline


async def run_equipment_extraction():
    """Example: Extract electronic equipment specifications"""
    
    equipment_prompt = """
Extract electronic equipment information from this document:

- manufacturer: Company name (Apple, Microsoft, Lenovo, Acer, etc.)
- year: Product year or document year (2020, 2021, 2022, 2023, 2024)
- product_name: Product model or name
- carbon_footprint: CO2 emissions or carbon footprint (kg CO2 or CO2eq)
- power_consumption: Electrical consumption (Watts, W, kWh)
- weight: Product weight (kg, grams)

Return ONLY valid JSON with these exact field names. Use null for missing information.
Look through the entire document for this information.
"""
    
    pipeline = PDFPipeline()
    await pipeline.initialize()
    
    # Process PDFs in data folder
    data_folder = Path(__file__).parent / "data"
    results = await pipeline.run_batch_processing(
        pdf_folder=str(data_folder),
        custom_prompt=equipment_prompt,
        output_file="equipment_extraction_results.json"
    )
    
    # Export to CSV
    pipeline.export_to_csv("equipment_extraction_results.csv")
    
    # Show summary
    summary = pipeline.get_extraction_summary()
    print(f"\n📊 EXTRACTION SUMMARY")
    print(f"Success rate: {summary['success_rate']:.1f}%")
    print(f"Average processing time: {summary['average_processing_time']:.2f}s")
    print(f"Field extraction rates:")
    for field, rate in summary['field_extraction_rates'].items():
        print(f"  - {field}: {rate:.1f}%")
    
    return results


async def run_custom_extraction(prompt: str, output_name: str = "custom"):
    """Run custom extraction with user-provided prompt"""
    
    pipeline = PDFPipeline()
    await pipeline.initialize()
    
    data_folder = Path(__file__).parent / "data"
    results = await pipeline.run_batch_processing(
        pdf_folder=str(data_folder),
        custom_prompt=prompt,
        output_file=f"{output_name}_results.json"
    )
    
    pipeline.export_to_csv(f"{output_name}_results.csv")
    
    summary = pipeline.get_extraction_summary()
    print(f"\n📊 EXTRACTION SUMMARY")
    print(f"Success rate: {summary['success_rate']:.1f}%")
    print(f"Fields extracted: {list(summary['extracted_fields'].keys())}")
    
    return results


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="PDF Processing Pipeline Runner")
    parser.add_argument("--mode", choices=["equipment", "custom"], default="equipment",
                       help="Processing mode")
    parser.add_argument("--prompt", type=str, help="Custom extraction prompt (for custom mode)")
    parser.add_argument("--output", type=str, default="results", 
                       help="Output file prefix")
    
    args = parser.parse_args()
    
    if args.mode == "equipment":
        print("🔧 Running equipment extraction pipeline...")
        asyncio.run(run_equipment_extraction())
    
    elif args.mode == "custom":
        if not args.prompt:
            print("❌ Custom mode requires --prompt argument")
            return
        
        print(f"🔧 Running custom extraction pipeline...")
        print(f"Prompt: {args.prompt[:100]}...")
        asyncio.run(run_custom_extraction(args.prompt, args.output))


if __name__ == "__main__":
    main()