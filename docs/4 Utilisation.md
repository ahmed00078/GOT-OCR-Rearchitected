## 4. Utilisation

### 4.1 Interface Web (Frontend)

**Accès** : http://localhost:8000

#### OCR Simple
1. **Upload** : Glissez une image dans la zone de dépôt
2. **Mode** : Laissez "Enable AI reasoning" décochée
3. **Traitement** : Cliquez "Process"
4. **Résultat** : Texte extrait affiché

#### Extraction IA
1. **Upload** : Glissez une image dans la zone de dépôt
2. **Mode** : Cochez "Enable AI reasoning"
3. **Instructions** : Tapez vos instructions (ex: "extract contact info")
4. **Traitement** : Cliquez "Process"
5. **Résultat** : Données structurées + temps de traitement

### 4.2 CLI (Command Line)

#### OCR Simple
```bash
python cli.py ocr image.jpg
```

#### Extraction IA
```bash
python cli.py smart image.jpg "extract contacts and emails"
```

#### Configuration
```bash
python cli.py config --model "Qwen/Qwen3-8B"
```

### 4.3 API REST

#### OCR Simple
```bash
curl -X POST "http://localhost:8000/process" \
  -F "task=ocr" \
  -F "images=@image.jpg" \
  -F "enable_reasoning=false"
```

#### Extraction IA
```bash
curl -X POST "http://localhost:8000/process" \
  -F "task=ocr" \
  -F "images=@image.jpg" \
  -F "enable_reasoning=true" \
  -F "custom_instructions=extract product name and price"
```

### Formats de Sortie

#### OCR Simple
```json
{
  "text": "Texte extrait...",
  "processing_time": 2.3,
  "reasoning_enabled": false
}
```

#### Extraction IA
```json
{
  "text": "Texte extrait...",
  "extraction_result": {
    "extracted_data": {
      "product_name": "iPhone 15",
      "price": "999€"
    },
    "processing_time": 3.1
  },
  "reasoning_enabled": true
}
```