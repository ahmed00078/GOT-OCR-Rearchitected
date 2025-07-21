## 2. Architecture

### Composants Principaux

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend Web  │    │   CLI Tool      │    │   API Client    │
│   (HTML/JS)     │    │   (cli.py)      │    │   (HTTP/JSON)   │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼───────────────┐
                    │       FastAPI Server        │
                    │         (main.py)           │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │   Enhanced OCR Service      │
                    │  (enhanced_ocr_service.py)  │
                    └─────────────┬───────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
   ┌────▼────┐              ┌─────▼─────┐         ┌────────▼────────┐
   │ OCR     │              │ Reasoning │         │ PDF Processing  │
   │ Service │              │ Service   │         │ Service         │
   │         │              │           │         │                 │
   └─────────┘              └───────────┘         └─────────────────┘
```

### Flow de Traitement

#### Mode OCR Simple
1. **Input** : Image/PDF uploadée
2. **OCR** : Extraction du texte brut
3. **Output** : Texte formaté

#### Mode IA Extraction
1. **Input** : Image/PDF + Instructions personnalisées
2. **OCR** : Extraction du texte brut
3. **IA** : Analyse et extraction de données spécifiques
4. **Output** : Données structurées (JSON)

### Services Backend

- **OCR Service** : Reconnaissance de texte avec GOT-OCR
- **Reasoning Service** : Analyse intelligente avec modèles IA
- **PDF Service** : Conversion PDF vers images
- **Enhanced OCR Service** : Orchestration des services

### Points d'Entrée

- **Frontend** : `http://localhost:8000`
- **API** : `http://localhost:8000/process`
- **CLI** : `python cli.py`