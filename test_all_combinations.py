#!/usr/bin/env python3
"""
Script de test complet pour GOT-OCR 2.0 + SmolLM2
Tests toutes les combinaisons possibles : images, PDFs, OCR seul, OCR + IA
"""

import asyncio
import aiohttp
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional


class OCRTester:
    """Testeur automatique pour toutes les combinaisons OCR"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
        
        # Chemins des données de test
        self.images_dir = Path("data/images")
        self.pdfs_dir = Path("data/pdfs")
        
        # Définir tous les types de tests
        self.test_combinations = self._define_test_combinations()
    
    def _define_test_combinations(self) -> List[Dict[str, Any]]:
        """Définir toutes les combinaisons de test"""
        
        # Instructions personnalisées pour les tests IA
        ai_instructions = [
            "Extract all text content and identify any technical specifications, contact information, or key data points",
            "Find and extract contact information including emails, phones, and addresses",
            "Extract any numerical data, dates, percentages, and statistical information",
            "Identify and extract product information, prices, specifications, and technical details"
        ]
        
        combinations = []
        
        # === TESTS AVEC IMAGES ===
        for img_file in ["1.jpg", "layout.jpg", "recommended-pace.jpg"]:
            img_path = self.images_dir / img_file
            if img_path.exists():
                
                # 1. Plain Text OCR seul
                combinations.append({
                    "name": f"Image_{img_file}_PlainText",
                    "files": [str(img_path)],
                    "params": {
                        "task": "Plain Text OCR",
                        "enable_reasoning": False
                    }
                })
                
                # 2. Format Text OCR seul
                combinations.append({
                    "name": f"Image_{img_file}_Format",
                    "files": [str(img_path)],
                    "params": {
                        "task": "Format Text OCR",
                        "ocr_type": "format",
                        "enable_reasoning": False
                    }
                })
                
                # 3. Fine-grained OCR (Box) - utilise des coordonnées d'exemple
                combinations.append({
                    "name": f"Image_{img_file}_Box",
                    "files": [str(img_path)],
                    "params": {
                        "task": "Fine-grained OCR (Box)",
                        "ocr_box": "[100,100,400,300]",
                        "enable_reasoning": False
                    }
                })
                
                # 4. Fine-grained OCR (Color)
                for color in ["red", "green", "blue"]:
                    combinations.append({
                        "name": f"Image_{img_file}_Color_{color}",
                        "files": [str(img_path)],
                        "params": {
                            "task": "Fine-grained OCR (Color)",
                            "ocr_color": color,
                            "enable_reasoning": False
                        }
                    })
                
                # 5. Multi-crop OCR seul
                combinations.append({
                    "name": f"Image_{img_file}_MultiCrop",
                    "files": [str(img_path)],
                    "params": {
                        "task": "Multi-crop OCR",
                        "enable_reasoning": False
                    }
                })
                
                # 6. Plain Text OCR + IA
                combinations.append({
                    "name": f"Image_{img_file}_PlainText_AI",
                    "files": [str(img_path)],
                    "params": {
                        "task": "Plain Text OCR",
                        "enable_reasoning": True,
                        "custom_instructions": ai_instructions[0]
                    }
                })
                
                # 7. Format Text OCR + IA
                combinations.append({
                    "name": f"Image_{img_file}_Format_AI",
                    "files": [str(img_path)],
                    "params": {
                        "task": "Format Text OCR",
                        "ocr_type": "format",
                        "enable_reasoning": True,
                        "custom_instructions": ai_instructions[1]
                    }
                })
                
                # 8. Multi-crop OCR + IA
                combinations.append({
                    "name": f"Image_{img_file}_MultiCrop_AI",
                    "files": [str(img_path)],
                    "params": {
                        "task": "Multi-crop OCR",
                        "enable_reasoning": True,
                        "custom_instructions": ai_instructions[2]
                    }
                })
        
        # === TESTS AVEC PDFs ===
        pdf_files = ["CV.pdf", "SmolLM2_ When Smol Goes Big —.pdf"]
        for pdf_file in pdf_files:
            pdf_path = self.pdfs_dir / pdf_file
            if pdf_path.exists():
                
                # 9. Multi-page OCR seul
                combinations.append({
                    "name": f"PDF_{pdf_file.replace(' ', '_').replace('.pdf', '')}_MultiPage",
                    "files": [str(pdf_path)],
                    "params": {
                        "task": "Multi-page OCR",
                        "enable_reasoning": False
                    }
                })
                
                # 10. Multi-page OCR + Format
                combinations.append({
                    "name": f"PDF_{pdf_file.replace(' ', '_').replace('.pdf', '')}_MultiPage_Format",
                    "files": [str(pdf_path)],
                    "params": {
                        "task": "Multi-page OCR",
                        "ocr_type": "format",
                        "enable_reasoning": False
                    }
                })
                
                # 11. Multi-page OCR + IA
                combinations.append({
                    "name": f"PDF_{pdf_file.replace(' ', '_').replace('.pdf', '')}_MultiPage_AI",
                    "files": [str(pdf_path)],
                    "params": {
                        "task": "Multi-page OCR",
                        "enable_reasoning": True,
                        "custom_instructions": ai_instructions[3]
                    }
                })
                
                # 12. Multi-page OCR + Format + IA
                combinations.append({
                    "name": f"PDF_{pdf_file.replace(' ', '_').replace('.pdf', '')}_MultiPage_Format_AI",
                    "files": [str(pdf_path)],
                    "params": {
                        "task": "Multi-page OCR",
                        "ocr_type": "format",
                        "enable_reasoning": True,
                        "custom_instructions": ai_instructions[0]
                    }
                })
        
        # === TESTS MULTI-FICHIERS ===
        # 13. Plusieurs images + Multi-page OCR
        all_images = [str(self.images_dir / f) for f in ["1.jpg", "layout.jpg"] 
                     if (self.images_dir / f).exists()][:2]  # Limiter à 2 pour éviter timeout
        if len(all_images) >= 2:
            combinations.append({
                "name": "MultiImages_MultiPage",
                "files": all_images,
                "params": {
                    "task": "Multi-page OCR",
                    "enable_reasoning": False
                }
            })
            
            # 14. Plusieurs images + Multi-page OCR + IA
            combinations.append({
                "name": "MultiImages_MultiPage_AI",
                "files": all_images,
                "params": {
                    "task": "Multi-page OCR",
                    "enable_reasoning": True,
                    "custom_instructions": "Extract and compare key information from all images, identifying similarities and differences"
                }
            })
        
        return combinations
    
    async def check_health(self) -> bool:
        """Vérifier que le service est en ligne"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/health") as response:
                    if response.status == 200:
                        health_data = await response.json()
                        print(f"✅ Service en ligne - OCR: {health_data['ocr_model_loaded']}, IA: {health_data['reasoning_model_loaded']}")
                        return True
                    else:
                        print(f"❌ Service non disponible (status: {response.status})")
                        return False
        except Exception as e:
            print(f"❌ Erreur connexion au service: {e}")
            return False
    
    async def run_single_test(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Exécuter un test unique"""
        print(f"\n🧪 Test: {test_config['name']}")
        print(f"   📁 Fichiers: {len(test_config['files'])}")
        print(f"   ⚙️  Paramètres: {test_config['params']}")
        
        start_time = time.time()
        result = {
            "test_name": test_config['name'],
            "files_count": len(test_config['files']),
            "params": test_config['params'],
            "start_time": start_time,
            "success": False,
            "error": None,
            "response_data": None,
            "duration": 0
        }
        
        try:
            # Préparer les fichiers
            files = []
            for file_path in test_config['files']:
                if Path(file_path).exists():
                    files.append(('images', open(file_path, 'rb')))
                else:
                    raise FileNotFoundError(f"Fichier non trouvé: {file_path}")
            
            # Préparer les données du formulaire
            form_data = aiohttp.FormData()
            for key, value in test_config['params'].items():
                form_data.add_field(key, str(value))
            
            # Ajouter les fichiers
            for field_name, file_obj in files:
                form_data.add_field(field_name, file_obj)
            
            # Envoyer la requête
            timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes max
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(f"{self.base_url}/process", data=form_data) as response:
                    duration = time.time() - start_time
                    result["duration"] = duration
                    
                    if response.status == 200:
                        response_data = await response.json()
                        result["success"] = True
                        result["response_data"] = {
                            "text_length": len(response_data.get("text", "")),
                            "html_available": response_data.get("html_available", False),
                            "reasoning_enabled": response_data.get("reasoning_enabled", False),
                            "extraction_result": response_data.get("extraction_result"),
                            "performance_metrics": response_data.get("performance_metrics"),
                            "models_used": response_data.get("models_used")
                        }
                        print(f"   ✅ Succès ({duration:.2f}s) - Texte: {result['response_data']['text_length']} chars")
                        if result["response_data"]["reasoning_enabled"]:
                            confidence = response_data.get("extraction_result", {}).get("confidence", 0)
                            print(f"   🧠 IA activée - Confiance: {confidence:.3f}")
                    else:
                        error_text = await response.text()
                        result["error"] = f"HTTP {response.status}: {error_text}"
                        print(f"   ❌ Échec HTTP {response.status}")
            
            # Fermer les fichiers
            for _, file_obj in files:
                file_obj.close()
            
        except Exception as e:
            result["error"] = str(e)
            result["duration"] = time.time() - start_time
            print(f"   ❌ Erreur: {e}")
        
        return result
    
    async def run_all_tests(self) -> None:
        """Exécuter tous les tests"""
        print("🚀 Démarrage des tests complets GOT-OCR 2.0 + SmolLM2")
        print(f"📊 {len(self.test_combinations)} combinaisons à tester")
        
        # Vérifier la santé du service
        if not await self.check_health():
            print("❌ Service non disponible, arrêt des tests")
            return
        
        # Exécuter tous les tests
        start_time = time.time()
        for i, test_config in enumerate(self.test_combinations, 1):
            print(f"\n{'='*60}")
            print(f"Test {i}/{len(self.test_combinations)}")
            
            result = await self.run_single_test(test_config)
            self.results.append(result)
            
            # Pause courte entre les tests pour éviter la surchauffe
            await asyncio.sleep(2)
        
        total_duration = time.time() - start_time
        self.print_summary(total_duration)
        self.save_results()
    
    def print_summary(self, total_duration: float):
        """Afficher le résumé des tests"""
        successful = [r for r in self.results if r["success"]]
        failed = [r for r in self.results if not r["success"]]
        
        print(f"\n{'='*60}")
        print("📊 RÉSUMÉ DES TESTS")
        print(f"{'='*60}")
        print(f"⏱️  Durée totale: {total_duration:.2f}s")
        print(f"✅ Tests réussis: {len(successful)}/{len(self.results)}")
        print(f"❌ Tests échoués: {len(failed)}/{len(self.results)}")
        print(f"📈 Taux de réussite: {len(successful)/len(self.results)*100:.1f}%")
        
        if successful:
            avg_duration = sum(r["duration"] for r in successful) / len(successful)
            print(f"⏱️  Durée moyenne (succès): {avg_duration:.2f}s")
        
        # Détails des échecs
        if failed:
            print(f"\n❌ ÉCHECS DÉTAILLÉS:")
            for result in failed:
                print(f"   • {result['test_name']}: {result['error']}")
        
        # Statistiques par type
        print(f"\n📊 STATISTIQUES PAR TYPE:")
        ocr_only = [r for r in successful if not r["params"].get("enable_reasoning", False)]
        ocr_ai = [r for r in successful if r["params"].get("enable_reasoning", False)]
        
        print(f"   • OCR seul: {len(ocr_only)} tests réussis")
        print(f"   • OCR + IA: {len(ocr_ai)} tests réussis")
        
        if ocr_only:
            avg_ocr = sum(r["duration"] for r in ocr_only) / len(ocr_only)
            print(f"   • Temps moyen OCR: {avg_ocr:.2f}s")
        
        if ocr_ai:
            avg_ai = sum(r["duration"] for r in ocr_ai) / len(ocr_ai)
            print(f"   • Temps moyen OCR+IA: {avg_ai:.2f}s")
    
    def save_results(self):
        """Sauvegarder les résultats en JSON"""
        output_file = f"test_results_{int(time.time())}.json"
        
        # Nettoyer les résultats pour JSON
        clean_results = []
        for result in self.results:
            clean_result = result.copy()
            # Supprimer les données trop volumineuses
            if clean_result.get("response_data") and isinstance(clean_result["response_data"], dict):
                response_data = clean_result["response_data"].copy()
                # Garder seulement les métriques essentielles
                clean_result["response_data"] = {
                    "text_length": response_data.get("text_length"),
                    "html_available": response_data.get("html_available"),
                    "reasoning_enabled": response_data.get("reasoning_enabled"),
                    "performance_metrics": response_data.get("performance_metrics"),
                    "models_used": response_data.get("models_used")
                }
            clean_results.append(clean_result)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Résultats sauvegardés: {output_file}")


async def main():
    """Point d'entrée principal"""
    tester = OCRTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())