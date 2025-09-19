#!/usr/bin/env python3
"""
Test script for the improved pipeline
Compare results with the original pipeline
"""

import asyncio
import json
from pathlib import Path
from improved_pipeline import ImprovedPDFPipeline


async def test_improved_pipeline():
    """Test the improved pipeline and compare with previous results"""
    
    print("🧪 Testing Improved PDF Pipeline")
    print("=" * 60)
    
    # Initialize improved pipeline
    pipeline = ImprovedPDFPipeline()
    await pipeline.initialize()
    
    # Run on same data folder
    data_folder = Path(__file__).parent / "data"
    
    print("🔬 Running improved pipeline on test dataset...")
    results = await pipeline.run_batch_processing(
        pdf_folder=str(data_folder),
        output_file="improved_extraction_results.json"
    )
    
    # Compare with previous results if available
    old_results_file = "equipment_extraction_results.json"
    if Path(old_results_file).exists():
        print(f"\n📊 COMPARISON WITH PREVIOUS RESULTS")
        print("=" * 60)
        
        improvement_stats = pipeline.get_improvement_stats(old_results_file)
        
        if improvement_stats:
            print(f"📈 Success rate improvement:")
            print(f"  Previous: {improvement_stats['old_success_rate']:.1f}%")
            print(f"  Current:  {improvement_stats['new_success_rate']:.1f}%")
            print(f"  Change:   {improvement_stats['improvement']:+.1f} percentage points")
            
            print(f"\n🎯 Field extraction improvements:")
            for field, improvement in improvement_stats['field_improvements'].items():
                if improvement != 0:
                    print(f"  {field}: {improvement:+d} files")
    
    # Export comparison CSV
    pipeline.export_to_csv("improved_extraction_results.csv")
    
    # Generate summary
    summary = pipeline.get_extraction_summary()
    print(f"\n✨ FINAL SUMMARY")
    print(f"Success rate: {summary['success_rate']:.1f}%")
    print(f"Average processing time: {summary['average_processing_time']:.2f}s")
    
    return results


async def test_single_problematic_file():
    """Test on a single file that had issues before"""
    
    print("\n🔍 Testing single problematic file...")
    
    pipeline = ImprovedPDFPipeline()
    await pipeline.initialize()
    
    # Test on a file that had raw_response issues
    problematic_file = Path(__file__).parent / "data" / "pcf-ideapad-5-chromebook-14-itl6.pdf"
    
    if problematic_file.exists():
        result = await pipeline.process_single_pdf_with_retry(problematic_file)
        
        print(f"File: {result['filename']}")
        print(f"Success: {result['success']}")
        print(f"Document type: {result['document_type']}")
        print(f"Attempts: {result['attempt']}")
        print(f"Extracted data: {result['extracted_data']}")
        
        return result
    else:
        print("Problematic test file not found")
        return None


if __name__ == "__main__":
    print("🚀 Starting improved pipeline test...")
    
    # Run full test
    asyncio.run(test_improved_pipeline())
    
    # Test single problematic file
    asyncio.run(test_single_problematic_file())
    
    print("\n🎉 Test completed!")