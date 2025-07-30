#!/usr/bin/env python3
"""
Pipeline Utilities
Helper functions for data processing, analysis, and export
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class PipelineAnalyzer:
    """Analyze and visualize pipeline results"""
    
    def __init__(self, results_file: str):
        self.results_file = Path(results_file)
        self.results = self._load_results()
    
    def _load_results(self) -> List[Dict[str, Any]]:
        """Load results from JSON file"""
        if not self.results_file.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_file}")
        
        with open(self.results_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def generate_report(self, output_file: str = "pipeline_report.html"):
        """Generate comprehensive HTML report"""
        
        successful = [r for r in self.results if r["success"]]
        failed = [r for r in self.results if not r["success"]]
        
        # Calculate statistics
        total_files = len(self.results)
        success_rate = len(successful) / total_files * 100 if total_files > 0 else 0
        avg_processing_time = sum(r["processing_time"] for r in successful) / len(successful) if successful else 0
        total_processing_time = sum(r["processing_time"] for r in self.results)
        
        # Field analysis
        field_stats = self._analyze_fields()
        
        # Generate HTML report
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>PDF Pipeline Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
        .stat-box {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; text-align: center; }}
        .field-analysis {{ margin: 20px 0; }}
        .success {{ color: green; }}
        .error {{ color: red; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 PDF Processing Pipeline Report</h1>
        <p>Generated from: {self.results_file.name}</p>
        <p>Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="stats">
        <div class="stat-box">
            <h3>Total Files</h3>
            <h2>{total_files}</h2>
        </div>
        <div class="stat-box">
            <h3>Success Rate</h3>
            <h2 class="success">{success_rate:.1f}%</h2>
        </div>
        <div class="stat-box">
            <h3>Avg Processing Time</h3>
            <h2>{avg_processing_time:.2f}s</h2>
        </div>
        <div class="stat-box">
            <h3>Total Time</h3>
            <h2>{total_processing_time:.2f}s</h2>
        </div>
    </div>
    
    <div class="field-analysis">
        <h2>📋 Field Extraction Analysis</h2>
        <table>
            <tr><th>Field</th><th>Extracted Count</th><th>Success Rate</th></tr>
"""
        
        for field, stats in field_stats.items():
            success_rate_field = stats['count'] / total_files * 100
            html_content += f"""
            <tr>
                <td>{field}</td>
                <td>{stats['count']}</td>
                <td>{success_rate_field:.1f}%</td>
            </tr>
"""
        
        html_content += """
        </table>
    </div>
    
    <div class="detailed-results">
        <h2>📄 Detailed Results</h2>
        <table>
            <tr><th>Filename</th><th>Status</th><th>Processing Time</th><th>Text Length</th><th>Fields Extracted</th></tr>
"""
        
        for result in self.results:
            status_class = "success" if result["success"] else "error"
            status_text = "✅ Success" if result["success"] else f"❌ {result.get('error', 'Failed')}"
            fields_count = len([k for k, v in result.get("extracted_data", {}).items() 
                              if v and str(v).lower() not in ['null', 'none', '']])
            
            html_content += f"""
            <tr>
                <td>{result['filename']}</td>
                <td class="{status_class}">{status_text}</td>
                <td>{result['processing_time']:.2f}s</td>
                <td>{result.get('text_length', 0):,}</td>
                <td>{fields_count}</td>
            </tr>
"""
        
        html_content += """
        </table>
    </div>
</body>
</html>
"""
        
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📊 HTML report generated: {output_file}")
        return output_file
    
    def _analyze_fields(self) -> Dict[str, Dict[str, Any]]:
        """Analyze field extraction statistics"""
        field_stats = {}
        
        for result in self.results:
            if result["success"]:
                data = result.get("extracted_data", {})
                for field, value in data.items():
                    if field not in field_stats:
                        field_stats[field] = {"count": 0, "values": []}
                    
                    if value and str(value).lower() not in ['null', 'none', '']:
                        field_stats[field]["count"] += 1
                        field_stats[field]["values"].append(value)
        
        return field_stats
    
    def export_detailed_csv(self, output_file: str = "detailed_results.csv"):
        """Export detailed results with all extracted fields"""
        
        # Collect all unique fields
        all_fields = set()
        for result in self.results:
            if result["success"]:
                all_fields.update(result.get("extracted_data", {}).keys())
        
        # Create detailed rows
        rows = []
        for result in self.results:
            row = {
                "filename": result["filename"],
                "success": result["success"],
                "processing_time": result["processing_time"],
                "text_length": result.get("text_length", 0),
                "error": result.get("error", "")
            }
            
            # Add all possible fields
            extracted_data = result.get("extracted_data", {})
            for field in all_fields:
                row[f"extracted_{field}"] = extracted_data.get(field, "")
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
        print(f"📊 Detailed CSV exported: {output_file}")
        return output_file


class PromptTemplates:
    """Pre-defined prompt templates for common extraction tasks"""
    
    EQUIPMENT_SPECS = """
Extract electronic equipment information from this document:

- manufacturer: Company/brand name (Apple, Microsoft, Lenovo, etc.)
- year: Product year or document publication year
- product_name: Product model or full name
- carbon_footprint: CO2 emissions or carbon footprint (kg CO2)
- power_consumption: Electrical consumption (Watts, W)
- weight: Product weight (kg, grams)

Return ONLY valid JSON with these exact field names. Use null for missing data.
"""
    
    FINANCIAL_DATA = """
Extract financial information from this document:

- company: Company or organization name
- revenue: Annual revenue with currency
- operating_costs: Operating expenses
- net_profit: Net profit or loss
- currency: Primary currency used
- fiscal_year: Financial reporting year

Return ONLY valid JSON format. Use null for unavailable data.
"""
    
    CONTACT_INFO = """
Extract contact and organizational information:

- organization: Company or organization name
- email: Email addresses
- phone: Phone numbers
- address: Physical addresses
- website: Website URLs
- contact_person: Key contact names

Return ONLY valid JSON format. Use null for missing information.
"""
    
    SUSTAINABILITY_DATA = """
Extract sustainability and environmental information:

- company: Organization name
- carbon_emissions: CO2 emissions (tons, kg)
- energy_consumption: Energy usage (kWh, GWh)
- renewable_energy: Renewable energy percentage
- waste_reduction: Waste reduction metrics
- sustainability_goals: Environmental targets
- reporting_year: Sustainability report year

Return ONLY valid JSON format. Use null for unavailable data.
"""
    
    @classmethod
    def get_template(cls, template_name: str) -> str:
        """Get a specific template by name"""
        templates = {
            "equipment": cls.EQUIPMENT_SPECS,
            "financial": cls.FINANCIAL_DATA,
            "contact": cls.CONTACT_INFO,
            "sustainability": cls.SUSTAINABILITY_DATA
        }
        
        if template_name not in templates:
            available = list(templates.keys())
            raise ValueError(f"Template '{template_name}' not found. Available: {available}")
        
        return templates[template_name]
    
    @classmethod
    def list_templates(cls) -> Dict[str, str]:
        """List all available templates with descriptions"""
        return {
            "equipment": "Electronic equipment specifications and environmental data",
            "financial": "Financial data including revenue, costs, and profits",
            "contact": "Contact information and organizational details",
            "sustainability": "Environmental and sustainability metrics"
        }


def create_sample_config(output_file: str = "pipeline_config.json"):
    """Create a sample configuration file for the pipeline"""
    
    config = {
        "input_folder": "data",
        "output_folder": "results",
        "extraction_templates": {
            "equipment": PromptTemplates.EQUIPMENT_SPECS.strip(),
            "financial": PromptTemplates.FINANCIAL_DATA.strip(),
            "contact": PromptTemplates.CONTACT_INFO.strip(),
            "sustainability": PromptTemplates.SUSTAINABILITY_DATA.strip()
        },
        "processing_options": {
            "max_files": None,  # None for all files
            "timeout_seconds": 120,
            "export_formats": ["json", "csv", "html"]
        },
        "output_settings": {
            "include_full_text": False,
            "include_processing_stats": True,
            "generate_report": True
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Sample configuration created: {output_file}")
    return output_file


if __name__ == "__main__":
    print("🔧 Pipeline Utilities")
    print("Available functions:")
    print("- PipelineAnalyzer: Analyze pipeline results")
    print("- PromptTemplates: Pre-defined extraction prompts")
    print("- create_sample_config(): Generate configuration file")
    
    # Create sample config
    create_sample_config()
    
    # Show available templates
    print("\n📋 Available prompt templates:")
    for name, description in PromptTemplates.list_templates().items():
        print(f"  - {name}: {description}")