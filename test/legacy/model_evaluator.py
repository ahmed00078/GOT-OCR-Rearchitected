#!/usr/bin/env python3
"""
Simplified AI Model Tester for Environmental Data
Focus: Test models and save raw responses (no complex scoring)

Usage:
    python simple_model_tester.py --model ollama:qwen3:8b --quick
    python simple_model_tester.py --model transformers:microsoft/Phi-3.5-mini-instruct
    python simple_model_tester.py --compare-responses
"""

import json
import time
import argparse
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import re

# Only import what we need
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

@dataclass
class TestResult:
    """Simple test result - just responses"""
    document_id: int
    question_id: str
    model_name: str
    question: str
    predicted_answer: str
    expected_answer: Dict
    processing_time: float
    error: Optional[str] = None

class ModelInterface:
    """Base interface for different model types"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_loaded = False
        
    def load_model(self):
        """Load the model"""
        raise NotImplementedError
        
    def generate_response(self, prompt: str) -> str:
        """Generate response from prompt"""
        raise NotImplementedError
        
    def cleanup(self):
        """Cleanup resources"""
        pass

class OllamaInterface(ModelInterface):
    """Interface for Ollama models"""
    
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        super().__init__(model_name)
        self.base_url = base_url
        
    def load_model(self):
        """Check if Ollama is available and model exists"""            
        try:
            # Test Ollama connectivity
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code != 200:
                raise ConnectionError("Cannot connect to Ollama service")
                
            # Check if model is available
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            
            if self.model_name not in model_names:
                print(f"⚠️  Model {self.model_name} not found. Available models: {model_names}")
                print(f"🔄 Attempting to pull model...")
                
                # Try to pull the model
                pull_response = requests.post(
                    f"{self.base_url}/api/pull",
                    json={"name": self.model_name},
                    timeout=300
                )
                
                if pull_response.status_code != 200:
                    raise RuntimeError(f"Failed to pull model {self.model_name}")
                    
            self.is_loaded = True
            print(f"✅ Ollama model {self.model_name} ready")
            
        except Exception as e:
            raise RuntimeError(f"Ollama setup failed: {str(e)}")
    
    def generate_response(self, prompt: str) -> str:
        """Generate response using Ollama API"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "num_predict": 1024  # More tokens for complete responses
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            else:
                raise RuntimeError(f"Ollama API error: {response.status_code}")
                
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {str(e)}")

class TransformersInterface(ModelInterface):
    """Interface for Hugging Face Transformers models"""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.tokenizer = None
        self.model = None
        self.device = None
        
    def load_model(self):
        """Load Transformers model"""            
        try:
            print(f"🔄 Loading Transformers model: {self.model_name}")
            
            # Determine device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"📱 Using device: {self.device}")
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model.eval()
            self.is_loaded = True
            print(f"✅ Transformers model loaded")
            
        except Exception as e:
            raise RuntimeError(f"Transformers loading failed: {str(e)}")
    
    def generate_response(self, prompt: str) -> str:
        """Generate response using Transformers"""
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,  # More tokens for complete responses
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            raise RuntimeError(f"Transformers generation failed: {str(e)}")
    
    def cleanup(self):
        """Cleanup Transformers resources"""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class SimpleModelTester:
    """Simplified tester - just run models and save responses"""
    
    def __init__(self, dataset_path: str = "dataset.json"):
        self.dataset = self._load_dataset(dataset_path)
        
    def _load_dataset(self, path: str) -> Dict:
        """Load the evaluation dataset"""
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        else:
            # Use embedded dataset if file not found
            print(f"⚠️  Dataset file {path} not found, using embedded dataset")
            return self._get_embedded_dataset()
    
    def _get_embedded_dataset(self) -> Dict:
        """Return embedded dataset (subset for testing)"""
        return {
            "dataset_info": {"name": "Embedded Test Dataset", "total_documents": 2},
            "documents": [
                {
                    "id": 1,
                    "category": "carbon_footprint",
                    "difficulty": "easy",
                    "title": "HP Laptop Environmental Sheet",
                    "text": "HP EliteBook 840 G9 Environmental Data\n\nCarbon Footprint: 285 kg CO₂ equivalent\nPower consumption: 45W typical, 2.1W sleep\nCertifications: ENERGY STAR 8.0, EPEAT Gold",
                    "questions": [
                        {
                            "id": "q1_1",
                            "question": "What is the total carbon footprint?",
                            "expected_answer": {"carbon_footprint": {"total": {"value": 285, "unit": "kg CO₂ equivalent"}}}
                        }
                    ]
                },
                {
                    "id": 2,
                    "category": "technical_specs",
                    "difficulty": "medium",
                    "title": "Dell Monitor Technical Specifications",
                    "text": "Dell UltraSharp U2723QE 27\" Monitor\nPower consumption: 32W (100%), 24W (50%), 0.3W standby\nRecyclability: 78% by weight\nCertifications: ENERGY STAR 8.0, EPEAT Gold",
                    "questions": [
                        {
                            "id": "q2_1",
                            "question": "What are the power consumption modes?",
                            "expected_answer": {"power_modes": {"on_100": 32, "on_50": 24, "standby": 0.3}}
                        }
                    ]
                }
            ]
        }
    
    def create_model_interface(self, model_spec: str) -> ModelInterface:
        """Create appropriate model interface based on specification"""
        if model_spec.startswith("ollama:"):
            model_name = model_spec.split(":", 1)[1]
            return OllamaInterface(model_name)
        elif model_spec.startswith("transformers:"):
            model_name = model_spec.split(":", 1)[1]
            return TransformersInterface(model_name)
        else:
            raise ValueError(f"Unknown model specification: {model_spec}")
    
    def create_prompt(self, text: str, question: str) -> str:
        """Create simple prompt for extraction"""
        return f"""TASK: Extract ONLY the requested information from the electronics specification text.  /nothink and /no_think
                TEXT:
                {text}
                
                QUESTION: {question}
                
                Instructions:
                - Return a single valid JSON object only, with NO markdown formatting (no ```json or language tags)
                - Extract exact values with original units
                - Do NOT perform any calculations
                - Do NOT include information not specifically asked for
                - Use "null" for missing information
                - Numbers should be numeric values, units as separate strings
                
                FORMAT EXAMPLE:
                {{
                  "field_name": {{"value": 123, "unit": "W"}},
                  "text_field": "exact text from source"
                }}
                
                Response:
                """
    
    def test_model(
        self, 
        model_spec: str, 
        category_filter: Optional[str] = None,
        max_documents: Optional[int] = None
    ) -> List[TestResult]:
        """Test a model and return raw responses"""
        
        print(f"\n🧪 TESTING MODEL: {model_spec}")
        print("=" * 50)
        
        # Create and load model
        try:
            model = self.create_model_interface(model_spec)
            model.load_model()
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return []
        
        # Filter documents
        documents = self.dataset["documents"]
        if category_filter:
            documents = [d for d in documents if d["category"] == category_filter]
        if max_documents:
            documents = documents[:max_documents]
        
        print(f"📊 Testing on {len(documents)} documents")
        
        results = []
        
        for doc in documents:
            print(f"\n📄 Document {doc['id']}: {doc['title']}")
            print(f"   Category: {doc['category']}")
            
            for question_data in doc["questions"]:
                question_id = question_data["id"]
                question = question_data["question"]
                expected = question_data["expected_answer"]
                
                print(f"   ❓ {question_id}: {question}")
                
                # Generate prompt and get response
                prompt = self.create_prompt(doc["text"], question)
                
                start_time = time.time()
                try:
                    predicted = model.generate_response(prompt)
                    processing_time = time.time() - start_time
                    
                    result = TestResult(
                        document_id=doc["id"],
                        question_id=question_id,
                        model_name=model_spec,
                        question=question,
                        predicted_answer=predicted,
                        expected_answer=expected,
                        processing_time=processing_time
                    )
                    
                    print(f"      ⏱️  Time: {processing_time:.2f}s")
                    print(f"      📝 Response preview: {predicted[:100]}...")
                    
                except Exception as e:
                    processing_time = time.time() - start_time
                    result = TestResult(
                        document_id=doc["id"],
                        question_id=question_id,
                        model_name=model_spec,
                        question=question,
                        predicted_answer="",
                        expected_answer=expected,
                        processing_time=processing_time,
                        error=str(e)
                    )
                    
                    print(f"      ❌ Error: {e}")
                
                results.append(result)
        
        # Cleanup model
        model.cleanup()
        
        print(f"\n📊 SUMMARY")
        print(f"   Total questions: {len(results)}")
        print(f"   Successful: {len([r for r in results if r.error is None])}")
        print(f"   Average time: {sum(r.processing_time for r in results) / len(results):.2f}s")
        
        return results
    
    def save_responses(self, results: List[TestResult], output_dir: str = "responses"):
        """Save raw responses to file"""
        if not results:
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        model_name = results[0].model_name
        clean_name = re.sub(r'[^\w\-]', '_', model_name)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{clean_name}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Save in simple format
        data = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "total_questions": len(results),
            "successful": len([r for r in results if r.error is None]),
            "responses": [
                {
                    "document_id": r.document_id,
                    "question_id": r.question_id,
                    "question": r.question,
                    "model_response": r.predicted_answer,
                    "expected_structure": r.expected_answer,
                    "processing_time": r.processing_time,
                    "error": r.error
                }
                for r in results
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Responses saved to: {filepath}")
        return filepath
    
    def compare_responses(self, response_dir: str = "responses"):
        """Simple comparison of response files"""
        if not os.path.exists(response_dir):
            print("❌ No responses directory found")
            return
        
        files = [f for f in os.listdir(response_dir) if f.endswith('.json')]
        if len(files) < 2:
            print("❌ Need at least 2 response files for comparison")
            return
        
        print(f"\n📊 RESPONSE COMPARISON")
        print("=" * 40)
        
        for filename in sorted(files):
            filepath = os.path.join(response_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                model_name = data.get('model_name', 'Unknown')
                successful = data.get('successful', 0)
                total = data.get('total_questions', 0)
                
                print(f"📁 {filename}")
                print(f"   Model: {model_name}")
                print(f"   Success: {successful}/{total}")
                
                # Show first response as example
                if data.get('responses'):
                    first_response = data['responses'][0]
                    print(f"   Sample: {first_response['model_response'][:80]}...")
                print()
                
            except Exception as e:
                print(f"⚠️  Could not load {filename}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Simple AI Model Tester")
    parser.add_argument("--model", type=str, help="Model specification (e.g., ollama:qwen3:8b)")
    parser.add_argument("--category", type=str, help="Filter by category")
    parser.add_argument("--quick", action="store_true", help="Quick test (2 documents)")
    parser.add_argument("--compare-responses", action="store_true", help="Compare saved responses")
    parser.add_argument("--dataset", type=str, default="dataset.json", help="Dataset file")
    parser.add_argument("--output-dir", type=str, default="responses", help="Output directory")
    
    args = parser.parse_args()
    
    tester = SimpleModelTester(args.dataset)
    
    if args.compare_responses:
        tester.compare_responses(args.output_dir)
    
    elif args.model:
        max_docs = 2 if args.quick else None
        
        results = tester.test_model(
            model_spec=args.model,
            category_filter=args.category,
            max_documents=max_docs
        )
        
        if results:
            tester.save_responses(results, args.output_dir)
    
    else:
        print("❌ Please specify --model or --compare-responses")
        print("\nExample usage:")
        print("  python simple_model_tester.py --model ollama:qwen3:8b --quick")
        print("  python simple_model_tester.py --model transformers:microsoft/Phi-3.5-mini-instruct")
        print("  python simple_model_tester.py --compare-responses")

if __name__ == "__main__":
    main()