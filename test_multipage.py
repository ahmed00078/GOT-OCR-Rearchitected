# File: test_multipage.py
"""
Script de test complet pour la fonctionnalité Multi-Page
Tests automatisés pour validation des nouvelles fonctionnalités
"""

import asyncio
import os
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any

import requests
from PIL import Image, ImageDraw, ImageFont
import fitz  # PyMuPDF


class MultiPageTester:
    """Classe de test pour les fonctionnalités multi-page"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.test_results = []
        self.temp_dir = tempfile.mkdtemp()
        print(f"📁 Répertoire de test: {self.temp_dir}")
    
    def create_test_pdf(self, pages: int = 3, content_type: str = "text") -> str:
        """Créer un PDF de test avec du contenu spécifique"""
        pdf_path = os.path.join(self.temp_dir, f"test_{content_type}_{pages}pages.pdf")
        
        # Créer le PDF avec PyMuPDF
        doc = fitz.open()
        
        for i in range(pages):
            page = doc.new_page()
            
            if content_type == "text":
                # Texte simple
                text = f"""Page {i+1}
                
Cette est la page numéro {i+1} du document de test.

Contenu de test pour l'OCR:
- Élément {i+1}.1
- Élément {i+1}.2  
- Élément {i+1}.3

Texte additionnel pour tester la reconnaissance:
Lorem ipsum dolor sit amet, consectetur adipiscing elit.
Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.

Page {i+1} - Fin du contenu
"""
                
            elif content_type == "mixed":
                # Contenu mixte avec format
                text = f"""# Page {i+1} - Document Technique

## Section {i+1}.1 Introduction

Cette page contient du **texte formaté** pour tester l'OCR avancé.

### Formules mathématiques:
- E = mc²
- x² + y² = z²
- ∫ f(x)dx = F(x) + C

### Liste numérotée:
1. Premier élément 
2. Deuxième élément
3. Troisième élément

**Date**: 2025-05-27
**Page**: {i+1}/{pages}
"""
            
            elif content_type == "table":
                # Tableau de données
                text = f"""Page {i+1} - Données Tabulaires

| Nom       | Age | Ville      | Score |
|-----------|-----|------------|-------|
| Alice     | 25  | Paris      | 85.5  |
| Bob       | 30  | Lyon       | 92.1  |
| Charlie   | 35  | Marseille  | 78.3  |
| Diana     | 28  | Toulouse   | 94.7  |

Total des scores: 350.6
Moyenne: 87.65

Fin page {i+1}
"""
            
            # Écrire le texte sur la page
            page.insert_text((50, 50), text, fontsize=12)
        
        # Sauvegarder le PDF
        doc.save(pdf_path)
        doc.close()
        
        print(f"✅ PDF créé: {pdf_path} ({pages} pages)")
        return pdf_path
    
    def create_test_images(self, count: int = 3) -> List[str]:
        """Créer des images de test avec du texte"""
        image_paths = []
        
        for i in range(count):
            # Créer une image avec du texte
            img = Image.new('RGB', (800, 600), color='white')
            draw = ImageDraw.Draw(img)
            
            # Texte de test
            text = f"""Image {i+1}
            
Test OCR Image {i+1}

Contenu pour reconnaissance:
• Point {i+1}.1
• Point {i+1}.2
• Point {i+1}.3

Texte technique:
Model: GOT-OCR 2.0
Version: Multi-Page
Status: Test Image {i+1}
"""
            
            # Dessiner le texte (utilise une police par défaut)
            y_position = 50
            for line in text.strip().split('\n'):
                draw.text((50, y_position), line.strip(), fill='black')
                y_position += 25
            
            # Sauvegarder l'image
            img_path = os.path.join(self.temp_dir, f"test_image_{i+1}.png")
            img.save(img_path)
            image_paths.append(img_path)
        
        print(f"✅ Images créées: {len(image_paths)} fichiers")
        return image_paths
    
    def test_api_health(self) -> bool:
        """Tester la santé de l'API"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ API Health: {health_data}")
                return True
            else:
                print(f"❌ API Health failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ API non accessible: {e}")
            return False
    
    def test_pdf_processing(self, pdf_path: str, task: str = "Multi-page OCR") -> Dict[str, Any]:
        """Tester le traitement d'un PDF"""
        print(f"\n🧪 Test PDF: {os.path.basename(pdf_path)}")
        
        start_time = time.time()
        
        try:
            # Préparer la requête
            files = {'images': open(pdf_path, 'rb')}
            data = {
                'task': task,
                'ocr_type': 'format' if task in ['Format Text OCR', 'Multi-page OCR'] else None
            }
            
            # Envoyer la requête
            response = requests.post(
                f"{self.api_url}/process",
                files=files,
                data=data,
                timeout=180  # 3 minutes
            )
            
            files['images'].close()
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                test_result = {
                    "test_type": "PDF Processing",
                    "file": os.path.basename(pdf_path),
                    "task": task,
                    "status": "SUCCESS",
                    "processing_time": round(processing_time, 2),
                    "text_length": len(result.get('text', '')),
                    "html_available": result.get('html_available', False),
                    "multipage_info": result.get('multipage_info', {}),
                    "text_preview": result.get('text', '')[:200] + "..." if len(result.get('text', '')) > 200 else result.get('text', '')
                }
                
                print(f"✅ Succès: {processing_time:.2f}s")
                print(f"   Texte extrait: {len(result.get('text', ''))} caractères")
                print(f"   HTML disponible: {result.get('html_available', False)}")
                
                if result.get('multipage_info'):
                    info = result['multipage_info']
                    print(f"   Pages traitées: {info.get('total_pages', 'N/A')}")
                    print(f"   Méthode: {info.get('processing_method', 'N/A')}")
                
                return test_result
                
            else:
                error_result = {
                    "test_type": "PDF Processing",
                    "file": os.path.basename(pdf_path),
                    "task": task,
                    "status": "ERROR",
                    "processing_time": round(processing_time, 2),
                    "error": response.text
                }
                
                print(f"❌ Erreur {response.status_code}: {response.text}")
                return error_result
                
        except Exception as e:
            error_result = {
                "test_type": "PDF Processing",
                "file": os.path.basename(pdf_path),
                "task": task,
                "status": "EXCEPTION",
                "processing_time": round(time.time() - start_time, 2),
                "error": str(e)
            }
            
            print(f"❌ Exception: {e}")
            return error_result
    
    def test_mixed_files(self, pdf_path: str, image_paths: List[str]) -> Dict[str, Any]:
        """Tester le traitement de fichiers mixtes (PDF + images)"""
        print(f"\n🧪 Test Mixed Files: 1 PDF + {len(image_paths)} images")
        
        start_time = time.time()
        
        try:
            # Préparer les fichiers
            files = []
            files.append(('images', open(pdf_path, 'rb')))
            
            for img_path in image_paths[:2]:  # Limiter à 2 images pour le test
                files.append(('images', open(img_path, 'rb')))
            
            data = {
                'task': 'Multi-page OCR',
                'ocr_type': 'format'
            }
            
            # Envoyer la requête
            response = requests.post(
                f"{self.api_url}/process",
                files=files,
                data=data,
                timeout=300  # 5 minutes
            )
            
            # Fermer les fichiers
            for _, file in files:
                file.close()
            
            processing_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                test_result = {
                    "test_type": "Mixed Files",
                    "files": f"1 PDF + {len(image_paths[:2])} images",
                    "status": "SUCCESS",
                    "processing_time": round(processing_time, 2),
                    "text_length": len(result.get('text', '')),
                    "multipage_info": result.get('multipage_info', {})
                }
                
                print(f"✅ Succès mixed: {processing_time:.2f}s")
                if result.get('multipage_info'):
                    info = result['multipage_info']
                    print(f"   Total pages: {info.get('total_pages', 'N/A')}")
                    print(f"   PDFs: {info.get('pdf_count', 0)}, Images: {info.get('image_count', 0)}")
                
                return test_result
                
            else:
                print(f"❌ Erreur mixed files: {response.status_code}")
                return {
                    "test_type": "Mixed Files",
                    "status": "ERROR",
                    "processing_time": round(processing_time, 2),
                    "error": response.text
                }
                
        except Exception as e:
            print(f"❌ Exception mixed files: {e}")
            return {
                "test_type": "Mixed Files",
                "status": "EXCEPTION",
                "processing_time": round(time.time() - start_time, 2),
                "error": str(e)
            }
    
    def run_comprehensive_tests(self):
        """Exécuter tous les tests"""
        print("🚀 Démarrage des tests Multi-Page GOT-OCR 2.0")
        print("=" * 60)
        
        # Test 1: Santé de l'API
        if not self.test_api_health():
            print("❌ API non accessible - Tests annulés")
            return
        
        # Test 2: Créer les fichiers de test
        print("\n📝 Création des fichiers de test...")
        pdf_simple = self.create_test_pdf(3, "text")
        pdf_mixed = self.create_test_pdf(2, "mixed")
        pdf_table = self.create_test_pdf(1, "table")
        test_images = self.create_test_images(3)
        
        # Test 3: PDFs individuels
        print("\n📄 Tests de traitement PDF...")
        self.test_results.append(self.test_pdf_processing(pdf_simple, "Multi-page OCR"))
        self.test_results.append(self.test_pdf_processing(pdf_mixed, "Format Text OCR"))
        self.test_results.append(self.test_pdf_processing(pdf_table, "Multi-page OCR"))
        
        # Test 4: Images seules
        print("\n🖼️  Test images multiples...")
        image_test_result = self.test_mixed_files(test_images[0], test_images[1:])  # Utilise la première image comme "PDF"
        
        # Test 5: Fichiers mixtes
        print("\n🔀 Test fichiers mixtes...")
        mixed_result = self.test_mixed_files(pdf_simple, test_images[:2])
        self.test_results.append(mixed_result)
        
        # Résumé final
        self.print_summary()
    
    def print_summary(self):
        """Afficher le résumé des tests"""
        print("\n" + "=" * 60)
        print("📊 RÉSUMÉ DES TESTS")
        print("=" * 60)
        
        success_count = 0
        total_tests = len(self.test_results)
        
        for i, result in enumerate(self.test_results, 1):
            status_icon = "✅" if result["status"] == "SUCCESS" else "❌"
            print(f"{i}. {status_icon} {result['test_type']}")
            print(f"   Fichier: {result.get('file', result.get('files', 'N/A'))}")
            print(f"   Temps: {result['processing_time']}s")
            
            if result["status"] == "SUCCESS":
                success_count += 1
                if 'text_length' in result:
                    print(f"   Texte: {result['text_length']} caractères")
                if 'multipage_info' in result and result['multipage_info']:
                    info = result['multipage_info']
                    print(f"   Pages: {info.get('total_pages', 'N/A')}")
            else:
                print(f"   Erreur: {result.get('error', 'Inconnue')}")
            print()
        
        print(f"📈 Score de réussite: {success_count}/{total_tests} ({success_count/total_tests*100:.1f}%)")
        
        if success_count == total_tests:
            print("🎉 Tous les tests sont passés ! Multi-Page fonctionne correctement.")
        elif success_count > total_tests // 2:
            print("⚠️  La plupart des tests sont passés, mais il y a quelques problèmes.")
        else:
            print("🔧 Plusieurs tests ont échoué. Vérifiez la configuration.")
        
        # Nettoyage
        print(f"\n🧹 Nettoyage: {self.temp_dir}")


def main():
    """Fonction principale pour exécuter les tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Tests Multi-Page GOT-OCR 2.0")
    parser.add_argument("--api-url", default="http://localhost:8000", 
                       help="URL de l'API (défaut: http://localhost:8000)")
    parser.add_argument("--quick", action="store_true", 
                       help="Tests rapides seulement")
    
    args = parser.parse_args()
    
    tester = MultiPageTester(args.api_url)
    
    if args.quick:
        # Test rapide: juste un PDF simple
        print("⚡ Mode test rapide")
        tester.test_api_health()
        pdf_path = tester.create_test_pdf(2, "text")
        result = tester.test_pdf_processing(pdf_path)
        print(f"Résultat: {result['status']}")
    else:
        # Tests complets
        tester.run_comprehensive_tests()


if __name__ == "__main__":
    main()