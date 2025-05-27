# File: services/pdf_service.py
"""
Service de conversion PDF pour GOT-OCR 2.0
Gère la conversion des PDFs multi-pages en images pour l'OCR
"""

import logging
import os
import tempfile
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

import fitz  # PyMuPDF
from pdf2image import convert_from_path
from PIL import Image
from fastapi import UploadFile

logger = logging.getLogger(__name__)


class PDFService:
    """Service pour la conversion et le traitement des PDFs"""
    
    def __init__(self, config):
        self.config = config
        # Paramètres de conversion par défaut
        self.default_dpi = 300  # Qualité élevée pour l'OCR
        self.max_pages = 50     # Limite de sécurité
        self.output_format = "PNG"  # Format optimal pour l'OCR
    
    async def convert_pdf_to_images(
        self, 
        pdf_file: UploadFile, 
        temp_dir: str,
        dpi: int = None,
        max_pages: int = None
    ) -> List[str]:
        """
        Convertit un PDF en liste d'images
        
        Args:
            pdf_file: Fichier PDF uploadé
            temp_dir: Répertoire temporaire pour les images
            dpi: Résolution de conversion (défaut: 300)
            max_pages: Nombre max de pages (défaut: 50)
            
        Returns:
            Liste des chemins vers les images converties
            
        Raises:
            ValueError: Si le PDF est invalide ou trop volumineux
        """
        dpi = dpi or self.default_dpi
        max_pages = max_pages or self.max_pages
        
        # Sauvegarder temporairement le PDF
        pdf_path = os.path.join(temp_dir, f"temp_{pdf_file.filename}")
        
        try:
            # Écrire le PDF sur disque
            with open(pdf_path, "wb") as buffer:
                content = await pdf_file.read()
                buffer.write(content)
            
            # Vérifier le PDF avec PyMuPDF d'abord (plus rapide)
            pdf_info = self._validate_pdf(pdf_path, max_pages)
            logger.info(f"PDF validé: {pdf_info['pages']} pages, {pdf_info['size_mb']:.1f}MB")
            
            # Convertir avec pdf2image (meilleure qualité)
            images = convert_from_path(
                pdf_path,
                dpi=dpi,
                fmt=self.output_format.lower(),
                first_page=1,
                last_page=min(pdf_info['pages'], max_pages),
                thread_count=2  # Optimisation pour les gros PDFs
            )
            
            # Sauvegarder les images converties
            image_paths = []
            for i, image in enumerate(images):
                image_path = os.path.join(temp_dir, f"page_{i+1:03d}.png")
                image.save(image_path, "PNG", optimize=True)
                image_paths.append(image_path)
                logger.debug(f"Page {i+1} convertie: {image_path}")
            
            logger.info(f"PDF converti avec succès: {len(image_paths)} pages")
            return image_paths
            
        except Exception as e:
            logger.error(f"Erreur conversion PDF {pdf_file.filename}: {str(e)}")
            raise ValueError(f"Impossible de convertir le PDF: {str(e)}")
        
        finally:
            # Nettoyer le fichier PDF temporaire
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
    
    def _validate_pdf(self, pdf_path: str, max_pages: int) -> Dict[str, Any]:
        """
        Valide et analyse un PDF avec PyMuPDF
        
        Args:
            pdf_path: Chemin vers le PDF
            max_pages: Nombre maximum de pages autorisé
            
        Returns:
            Dictionnaire avec les informations du PDF
            
        Raises:
            ValueError: Si le PDF est invalide, protégé ou trop volumineux
        """
        try:
            # Ouvrir le PDF avec PyMuPDF
            doc = fitz.open(pdf_path)
            
            # Vérifications de base
            if doc.is_encrypted:
                doc.close()
                raise ValueError("PDF protégé par mot de passe non supporté")
            
            page_count = doc.page_count
            if page_count == 0:
                doc.close()
                raise ValueError("PDF vide ou corrompu")
            
            if page_count > max_pages:
                doc.close()
                raise ValueError(f"PDF trop volumineux: {page_count} pages (max: {max_pages})")
            
            # Informations sur le document
            metadata = doc.metadata
            file_size = os.path.getsize(pdf_path)
            
            doc.close()
            
            return {
                "pages": page_count,
                "size_bytes": file_size,
                "size_mb": file_size / (1024 * 1024),
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", "")
            }
            
        except fitz.FileDataError:
            raise ValueError("Fichier PDF corrompu ou invalide")
        except Exception as e:
            raise ValueError(f"Erreur lors de l'analyse du PDF: {str(e)}")
    
    async def extract_pdf_pages_advanced(
        self, 
        pdf_file: UploadFile, 
        temp_dir: str,
        page_range: Optional[Tuple[int, int]] = None
    ) -> List[str]:
        """
        Extraction avancée avec PyMuPDF pour de meilleures performances
        
        Args:
            pdf_file: Fichier PDF
            temp_dir: Répertoire temporaire
            page_range: Tuple (début, fin) pour extraire des pages spécifiques
            
        Returns:
            Liste des chemins vers les images extraites
        """
        pdf_path = os.path.join(temp_dir, f"temp_{pdf_file.filename}")
        
        try:
            # Sauvegarder le PDF
            with open(pdf_path, "wb") as buffer:
                content = await pdf_file.read()
                buffer.write(content)
            
            # Ouvrir avec PyMuPDF
            doc = fitz.open(pdf_path)
            
            # Déterminer les pages à extraire
            start_page = page_range[0] - 1 if page_range else 0
            end_page = page_range[1] if page_range else doc.page_count
            end_page = min(end_page, doc.page_count)
            
            image_paths = []
            
            for page_num in range(start_page, end_page):
                page = doc[page_num]
                
                # Convertir la page en image avec haute résolution
                mat = fitz.Matrix(300/72, 300/72)  # 300 DPI
                pix = page.get_pixmap(matrix=mat)
                
                # Sauvegarder l'image
                image_path = os.path.join(temp_dir, f"page_{page_num+1:03d}.png")
                pix.save(image_path)
                image_paths.append(image_path)
                
                logger.debug(f"Page {page_num+1} extraite: {image_path}")
            
            doc.close()
            logger.info(f"Extraction PyMuPDF terminée: {len(image_paths)} pages")
            return image_paths
            
        except Exception as e:
            logger.error(f"Erreur extraction PyMuPDF: {str(e)}")
            raise ValueError(f"Erreur d'extraction: {str(e)}")
        
        finally:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
    
    def get_pdf_info(self, pdf_path: str) -> Dict[str, Any]:
        """
        Obtient les informations détaillées d'un PDF
        
        Returns:
            Dictionnaire avec toutes les métadonnées du PDF
        """
        try:
            doc = fitz.open(pdf_path)
            
            info = {
                "page_count": doc.page_count,
                "metadata": doc.metadata,
                "is_encrypted": doc.is_encrypted,
                "pages_info": []
            }
            
            # Informations détaillées par page (première page seulement pour éviter la lenteur)
            if doc.page_count > 0:
                page = doc[0]
                page_info = {
                    "width": page.rect.width,
                    "height": page.rect.height,
                    "rotation": page.rotation,
                    "has_images": len(page.get_images()) > 0,
                    "has_text": len(page.get_text().strip()) > 0
                }
                info["pages_info"].append(page_info)
            
            doc.close()
            return info
            
        except Exception as e:
            logger.error(f"Erreur info PDF: {str(e)}")
            return {"error": str(e)}