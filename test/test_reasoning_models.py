# File: test_reasoning_models.py
"""
Script pour tester différents modèles de raisonnement sur le dataset environnemental
Usage: python test_reasoning_models.py --model phi-3.5 --test-id 1
"""

import json
import time
import argparse
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import re

# Exemple d'intégration avec différents modèles
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


@dataclass
class TestResult:
    """Résultat d'un test"""
    test_id: int
    model_name: str
    question: str
    predicted_answer: str
    expected_answer: Dict
    score: float
    processing_time: float
    error: Optional[str] = None


class ModelTester:
    """Testeur pour différents modèles de raisonnement"""
    
    def __init__(self):
        self.dataset = self._load_dataset()
        self.models = {}
        
    def _load_dataset(self) -> Dict:
        """Charger le dataset de test"""
        # Le dataset JSON que j'ai créé ci-dessus
        return {
            "test_dataset": [
                # Dataset complet ici...
                # Pour l'exemple, je mets juste le premier cas
                {
                    "id": 1,
                    "type": "laptop_simple",
                    "text": "HP EliteBook 840 G9 Specifications\nPower Consumption: 45W typical, 2.1W sleep mode\nCarbon Footprint: 285 kg CO2 equivalent\nRecyclability: 85% by weight\nCertifications: ENERGY STAR 8.0, EPEAT Gold\nWeight: 1.36 kg\nDimensions: 321 x 213 x 19.9 mm",
                    "questions": [
                        {
                            "question": "What is the product name and model?",
                            "expected_answer": {
                                "product_name": "HP EliteBook 840 G9",
                                "brand": "HP",
                                "model": "EliteBook 840 G9"
                            }
                        },
                        {
                            "question": "What are the power consumption values?",
                            "expected_answer": {
                                "typical_consumption": {"value": 45, "unit": "W"},
                                "sleep_consumption": {"value": 2.1, "unit": "W"}
                            }
                        }
                    ]
                }
            ]
        }
    
    def load_model(self, model_name: str):
        """Charger un modèle spécifique"""
        if model_name == "phi-3.5":
            return self._load_phi35()
        elif model_name == "qwen2.5":
            return self._load_qwen25()
        elif model_name == "gpt-3.5":
            return self._load_gpt35()
        elif model_name == "pattern-based":
            return self._load_pattern_based()
        else:
            raise ValueError(f"Modèle non supporté: {model_name}")
    
    def _load_phi35(self):
        """Charger Phi-3.5 Mini"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers requis pour Phi-3.5")
        
        model_name = "microsoft/Phi-3.5-mini-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        return {"tokenizer": tokenizer, "model": model, "type": "transformers"}
    
    def _load_qwen25(self):
        """Charger Qwen2.5-3B"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers requis pour Qwen2.5")
        
        model_name = "Qwen/Qwen2.5-3B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        return {"tokenizer": tokenizer, "model": model, "type": "transformers"}
    
    def _load_gpt35(self):
        """Charger GPT-3.5 (API)"""
        if not OPENAI_AVAILABLE:
            raise ImportError("openai requis pour GPT-3.5")
        return {"type": "openai", "model": "gpt-3.5-turbo"}
    
    def _load_pattern_based(self):
        """Modèle basé sur des patterns (baseline)"""
        return {"type": "pattern"}
    
    def generate_prompt(self, text: str, question: str) -> str:
        """Générer le prompt pour l'extraction"""
        return f"""You are an expert in analyzing technical documents for electronic equipment.

TEXT TO ANALYZE:
{text}

QUESTION: {question}

Please extract the requested information and respond in JSON format only. Be precise with numbers and units.

RESPONSE (JSON only):
"""

    def query_model(self, model_info: Dict, text: str, question: str) -> str:
        """Interroger un modèle"""
        if model_info["type"] == "transformers":
            return self._query_transformers_model(model_info, text, question)
        elif model_info["type"] == "openai":
            return self._query_openai_model(model_info, text, question)
        elif model_info["type"] == "pattern":
            return self._query_pattern_model(text, question)
        else:
            raise ValueError(f"Type de modèle non supporté: {model_info['type']}")
    
    def _query_transformers_model(self, model_info: Dict, text: str, question: str) -> str:
        """Interroger un modèle Transformers"""
        prompt = self.generate_prompt(text, question)
        
        tokenizer = model_info["tokenizer"]
        model = model_info["model"]
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    def _query_openai_model(self, model_info: Dict, text: str, question: str) -> str:
        """Interroger GPT-3.5 via API"""
        prompt = self.generate_prompt(text, question)
        
        response = openai.ChatCompletion.create(
            model=model_info["model"],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip()
    
    def _query_pattern_model(self, text: str, question: str) -> str:
        """Modèle baseline basé sur des patterns"""
        # Patterns simples pour extraction
        patterns = {
            "power": r'(\d+(?:\.\d+)?)\s*W',
            "co2": r'(\d+(?:\.\d+)?)\s*kg\s*CO2',
            "percentage": r'(\d+(?:\.\d+)?)\s*%',
            "product_name": r'^([A-Z][A-Za-z0-9\s]+(?:G\d+|Pro|Max|Ultra)?)',
        }
        
        results = {}
        for category, pattern in patterns.items():
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            if matches:
                results[category] = matches
        
        return json.dumps(results, indent=2)
    
    def evaluate_answer(self, predicted: str, expected: Dict) -> float:
        """Évaluer la qualité de la réponse"""
        try:
            # Extraire JSON de la réponse
            json_match = re.search(r'\{.*\}', predicted, re.DOTALL)
            if json_match:
                predicted_json = json.loads(json_match.group())
            else:
                return 0.0
            
            # Comparer avec la réponse attendue
            score = self._calculate_similarity(predicted_json, expected)
            return score
            
        except json.JSONDecodeError:
            return 0.0
        except Exception:
            return 0.0
    
    def _calculate_similarity(self, predicted: Dict, expected: Dict) -> float:
        """Calculer la similarité entre deux dictionnaires"""
        total_keys = len(expected)
        if total_keys == 0:
            return 1.0
        
        correct_keys = 0
        
        for key, expected_value in expected.items():
            if key in predicted:
                if isinstance(expected_value, dict):
                    if isinstance(predicted[key], dict):
                        # Comparaison récursive pour les dictionnaires imbriqués
                        nested_score = self._calculate_similarity(predicted[key], expected_value)
                        correct_keys += nested_score
                    else:
                        correct_keys += 0
                elif isinstance(expected_value, list):
                    if isinstance(predicted[key], list):
                        # Comparaison de listes
                        overlap = len(set(expected_value) & set(predicted[key]))
                        correct_keys += overlap / len(expected_value)
                    else:
                        correct_keys += 0
                else:
                    # Comparaison de valeurs simples
                    if str(predicted[key]).lower() == str(expected_value).lower():
                        correct_keys += 1
                    elif isinstance(expected_value, (int, float)):
                        try:
                            pred_num = float(predicted[key])
                            if abs(pred_num - expected_value) / expected_value < 0.1:  # 10% de tolérance
                                correct_keys += 1
                        except:
                            pass
        
        return correct_keys / total_keys
    
    def run_test(self, model_name: str, test_id: Optional[int] = None) -> List[TestResult]:
        """Exécuter les tests"""
        print(f"🧪 Test du modèle: {model_name}")
        
        # Charger le modèle
        try:
            model_info = self.load_model(model_name)
            print(f"✅ Modèle {model_name} chargé")
        except Exception as e:
            print(f"❌ Erreur chargement modèle: {e}")
            return []
        
        results = []
        test_cases = self.dataset["test_dataset"]
        
        # Filtrer par test_id si spécifié
        if test_id is not None:
            test_cases = [tc for tc in test_cases if tc["id"] == test_id]
        
        for test_case in test_cases:
            print(f"\n📄 Test {test_case['id']}: {test_case['type']}")
            
            for i, question_data in enumerate(test_case["questions"]):
                question = question_data["question"]
                expected = question_data["expected_answer"]
                
                print(f"  ❓ Question {i+1}: {question[:50]}...")
                
                # Mesurer le temps
                start_time = time.time()
                
                try:
                    # Interroger le modèle
                    predicted = self.query_model(model_info, test_case["text"], question)
                    processing_time = time.time() - start_time
                    
                    # Évaluer la réponse
                    score = self.evaluate_answer(predicted, expected)
                    
                    result = TestResult(
                        test_id=test_case["id"],
                        model_name=model_name,
                        question=question,
                        predicted_answer=predicted,
                        expected_answer=expected,
                        score=score,
                        processing_time=processing_time
                    )
                    
                    print(f"    📊 Score: {score:.2f} ({processing_time:.2f}s)")
                    
                except Exception as e:
                    result = TestResult(
                        test_id=test_case["id"],
                        model_name=model_name,
                        question=question,
                        predicted_answer="",
                        expected_answer=expected,
                        score=0.0,
                        processing_time=0.0,
                        error=str(e)
                    )
                    print(f"    ❌ Erreur: {e}")
                
                results.append(result)
        
        return results
    
    def generate_report(self, results: List[TestResult]) -> Dict:
        """Générer un rapport de performance"""
        if not results:
            return {}
        
        model_name = results[0].model_name
        total_tests = len(results)
        successful_tests = len([r for r in results if r.error is None])
        avg_score = sum(r.score for r in results) / total_tests
        avg_time = sum(r.processing_time for r in results) / total_tests
        
        # Scores par type de test
        test_type_scores = {}
        for result in results:
            test_data = next(tc for tc in self.dataset["test_dataset"] if tc["id"] == result.test_id)
            test_type = test_data["type"]
            if test_type not in test_type_scores:
                test_type_scores[test_type] = []
            test_type_scores[test_type].append(result.score)
        
        type_averages = {k: sum(v)/len(v) for k, v in test_type_scores.items()}
        
        report = {
            "model": model_name,
            "summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "success_rate": successful_tests / total_tests,
                "average_score": avg_score,
                "average_time": avg_time
            },
            "by_test_type": type_averages,
            "detailed_results": [
                {
                    "test_id": r.test_id,
                    "question": r.question[:100],
                    "score": r.score,
                    "time": r.processing_time,
                    "error": r.error
                }
                for r in results
            ]
        }
        
        return report


def main():
    parser = argparse.ArgumentParser(description="Test des modèles de raisonnement")
    parser.add_argument("--model", required=True, 
                       choices=["phi-3.5", "qwen2.5", "gpt-3.5", "pattern-based"],
                       help="Modèle à tester")
    parser.add_argument("--test-id", type=int, help="ID de test spécifique")
    parser.add_argument("--output", help="Fichier de sortie pour le rapport JSON")
    
    args = parser.parse_args()
    
    tester = ModelTester()
    results = tester.run_test(args.model, args.test_id)
    report = tester.generate_report(results)
    
    print("\n" + "="*60)
    print("📊 RAPPORT FINAL")
    print("="*60)
    print(f"Modèle: {report['model']}")
    print(f"Tests réussis: {report['summary']['successful_tests']}/{report['summary']['total_tests']}")
    print(f"Score moyen: {report['summary']['average_score']:.2f}")
    print(f"Temps moyen: {report['summary']['average_time']:.2f}s")
    
    print("\n📈 Scores par type de test:")
    for test_type, score in report['by_test_type'].items():
        print(f"  {test_type}: {score:.2f}")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n💾 Rapport sauvegardé: {args.output}")


if __name__ == "__main__":
    main()