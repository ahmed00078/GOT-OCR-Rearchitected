#!/usr/bin/env python3
"""
CLI Simple pour GOT-OCR 2.0 + IA
Usage:
  python cli.py ocr image.jpg                           # OCR simple
  python cli.py smart image.jpg "extract contacts"     # OCR + IA
  python cli.py config --model "Qwen/Qwen3-8B"         # Configuration
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional

# Import des services existants
from config import Config
from models.ocr_model import OCRModelManager
from services.enhanced_ocr_service import EnhancedOCRService
from fastapi import UploadFile, BackgroundTasks
import io


class SimpleCLI:
    """CLI minimal pour GOT-OCR"""
    
    def __init__(self):
        self.config = Config()
        self.model_manager = None
        self.ocr_service = None
        
    async def init_services(self):
        """Initialiser les services"""
        print("🔄 Initialisation des services...")
        
        # Initialiser le modèle OCR
        self.model_manager = OCRModelManager(self.config)
        await self.model_manager.load_model()
        
        # Initialiser le service OCR amélioré
        self.ocr_service = EnhancedOCRService(self.model_manager, self.config)
        
        # Initialiser le service IA si activé
        if self.config.reasoning_enabled:
            await self.ocr_service.initialize_reasoning()
        
        print("✅ Services initialisés")
    
    async def ocr_command(self, files: List[str], task: str = "Multi-page OCR", 
                         output_file: Optional[str] = None):
        """Commande OCR simple"""
        print(f"📄 OCR: {task}")
        print(f"📁 Fichiers: {files}")
        
        # Convertir les fichiers en UploadFile
        upload_files = []
        for file_path in files:
            if not Path(file_path).exists():
                print(f"❌ Fichier non trouvé: {file_path}")
                return
            
            with open(file_path, 'rb') as f:
                content = f.read()
                upload_file = UploadFile(
                    filename=Path(file_path).name,
                    file=io.BytesIO(content)
                )
                upload_files.append(upload_file)
        
        # Traitement OCR
        bg_tasks = BackgroundTasks()
        result = await self.ocr_service._process_with_layout(
            task=task,
            images=upload_files,
            background_tasks=bg_tasks
        )
        
        # Afficher le résultat
        print(f"\n📊 RÉSULTAT:")
        print(f"✅ Texte extrait ({len(result['text'])} caractères)")
        print(f"📄 Texte:\n{result['text'][:500]}...")
        
        # Sauvegarder si demandé
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                if output_file.endswith('.json'):
                    json.dump(result, f, indent=2, ensure_ascii=False)
                else:
                    f.write(result['text'])
            print(f"💾 Résultat sauvé: {output_file}")
    
    async def smart_command(self, files: List[str], instructions: str, 
                           output_file: Optional[str] = None):
        """Commande OCR + IA"""
        print(f"🧠 OCR + IA")
        print(f"📁 Fichiers: {files}")
        print(f"📋 Instructions: {instructions}")
        
        if not self.config.reasoning_enabled:
            print("❌ Le raisonnement IA n'est pas activé")
            return
        
        # Convertir les fichiers en UploadFile
        upload_files = []
        for file_path in files:
            if not Path(file_path).exists():
                print(f"❌ Fichier non trouvé: {file_path}")
                return
            
            with open(file_path, 'rb') as f:
                content = f.read()
                upload_file = UploadFile(
                    filename=Path(file_path).name,
                    file=io.BytesIO(content)
                )
                upload_files.append(upload_file)
        
        # Traitement OCR + IA
        bg_tasks = BackgroundTasks()
        result = await self.ocr_service.process_with_reasoning(
            task="Multi-page OCR",
            images=upload_files,
            background_tasks=bg_tasks,
            extraction_type="custom",
            custom_instructions=instructions
        )
        
        # Afficher le résultat
        print(f"\n📊 RÉSULTAT:")
        print(f"✅ Texte extrait ({len(result['text'])} caractères)")
        print(f"🧠 IA activée: {result['reasoning_enabled']}")
        
        if result.get('extraction_result'):
            extraction = result['extraction_result']
            print(f"🎯 Confiance: {extraction['confidence']:.2f}")
            print(f"📊 Données extraites:")
            print(json.dumps(extraction['extracted_data'], indent=2, ensure_ascii=False))
        
        # Sauvegarder si demandé
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                if output_file.endswith('.json'):
                    json.dump(result, f, indent=2, ensure_ascii=False)
                else:
                    f.write(result['text'])
            print(f"💾 Résultat sauvé: {output_file}")
    
    def config_command(self, model: Optional[str] = None, temperature: Optional[float] = None,
                      show: bool = False):
        """Commande de configuration"""
        if show:
            print("⚙️ Configuration actuelle:")
            print(f"🤖 Modèle OCR: {self.config.MODEL_NAME}")
            print(f"🧠 Modèle IA: {self.config.REASONING_MODEL_NAME}")
            print(f"🌡️ Température: {self.config.REASONING_TEMPERATURE}")
            print(f"🔧 IA activée: {self.config.ENABLE_REASONING}")
            return
        
        if model:
            print(f"🤖 Changement de modèle: {model}")
            print("⚠️  Modifiez manuellement config.py ligne 33:")
            print(f'REASONING_MODEL_NAME: str = "{model}"')
        
        if temperature:
            print(f"🌡️ Changement de température: {temperature}")
            print("⚠️  Modifiez manuellement config.py ligne 35:")
            print(f'REASONING_TEMPERATURE: float = {temperature}')


async def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(description="CLI pour GOT-OCR 2.0 + IA")
    subparsers = parser.add_subparsers(dest='command', help='Commandes disponibles')
    
    # Commande OCR
    ocr_parser = subparsers.add_parser('ocr', help='OCR simple')
    ocr_parser.add_argument('files', nargs='+', help='Fichiers à traiter')
    ocr_parser.add_argument('--task', default='Multi-page OCR', 
                           choices=['Plain Text OCR', 'Format Text OCR', 'Multi-page OCR'],
                           help='Type de tâche OCR')
    ocr_parser.add_argument('--output', help='Fichier de sortie')
    
    # Commande IA
    smart_parser = subparsers.add_parser('smart', help='OCR + IA')
    smart_parser.add_argument('files', nargs='+', help='Fichiers à traiter')
    smart_parser.add_argument('instructions', help='Instructions d\'extraction IA')
    smart_parser.add_argument('--output', help='Fichier de sortie')
    
    # Commande config
    config_parser = subparsers.add_parser('config', help='Configuration')
    config_parser.add_argument('--model', help='Modèle IA à utiliser')
    config_parser.add_argument('--temperature', type=float, help='Température du modèle')
    config_parser.add_argument('--show', action='store_true', help='Afficher la config')
    
    # Parser les arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialiser le CLI
    cli = SimpleCLI()
    
    # Traiter les commandes
    if args.command == 'config':
        cli.config_command(model=args.model, temperature=args.temperature, show=args.show)
    else:
        # Initialiser les services pour OCR/IA
        await cli.init_services()
        
        if args.command == 'ocr':
            await cli.ocr_command(files=args.files, task=args.task, output_file=args.output)
        elif args.command == 'smart':
            await cli.smart_command(files=args.files, instructions=args.instructions, 
                                  output_file=args.output)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Arrêt du CLI")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        sys.exit(1)