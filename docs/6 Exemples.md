## 6. Exemples Pratiques

### Cas d'Usage #1 : Extraction de Factures

#### CLI
```bash
python cli.py smart invoice.jpg "extract invoice number, date, total amount, supplier name"
```

#### API
```bash
curl -X POST "http://localhost:8000/process" \
  -F "task=ocr" \
  -F "images=@invoice.jpg" \
  -F "enable_reasoning=true" \
  -F "custom_instructions=extract invoice number, date, total amount, supplier name and format as JSON"
```

#### Frontend
1. Upload `invoice.jpg`
2. Cochez "Enable AI reasoning"
3. Instructions: `extract invoice number, date, total amount, supplier name`

**Résultat attendu** :
```json
{
  "invoice_number": "INV-2024-001",
  "date": "2024-01-15",
  "total_amount": "299.99€",
  "supplier_name": "Tech Solutions SARL"
}
```

### Cas d'Usage #2 : Extraction de Contacts

#### CLI
```bash
python cli.py smart business_card.png "extract all contact information"
```

#### API Python
```python
import requests

files = {'images': open('business_card.png', 'rb')}
data = {
    'task': 'ocr',
    'enable_reasoning': True,
    'custom_instructions': 'extract name, email, phone, company, position'
}

response = requests.post('http://localhost:8000/process', files=files, data=data)
result = response.json()
```

**Résultat attendu** :
```json
{
  "name": "Jean Dupont",
  "email": "jean.dupont@company.com",
  "phone": "+33 1 23 45 67 89",
  "company": "Innovation Corp",
  "position": "Directeur Technique"
}
```

### Cas d'Usage #3 : Analyse de Documents Administratifs

#### CLI Script
```bash
#!/bin/bash
# Traitement par lots
for file in documents/*.pdf; do
    python cli.py smart "$file" "extract document type, reference number, date"
done
```

#### API avec Retry
```python
import requests
import time

def process_document(filepath, instructions):
    for attempt in range(3):
        try:
            files = {'images': open(filepath, 'rb')}
            data = {
                'task': 'ocr',
                'enable_reasoning': True,
                'custom_instructions': instructions
            }
            
            response = requests.post('http://localhost:8000/process', 
                                   files=files, data=data, timeout=60)
            return response.json()
            
        except requests.exceptions.Timeout:
            if attempt < 2:
                time.sleep(5)
                continue
            raise
```

### Cas d'Usage #4 : Extraction de Produits E-commerce

#### Instructions Détaillées
```python
instructions = """
Extract product information with the following fields:
- product_name: The main product title
- price: Numerical price with currency
- description: Product description
- specifications: Technical details as list
- availability: In stock status
- rating: Customer rating if visible

Format the output as clean JSON.
"""
```

#### Résultat Structuré
```json
{
  "product_name": "Smartphone XYZ Pro",
  "price": "799.99€",
  "description": "Smartphone haut de gamme avec écran OLED",
  "specifications": [
    "Écran 6.7 pouces OLED",
    "128GB stockage",
    "Caméra 48MP"
  ],
  "availability": "En stock",
  "rating": "4.5/5"
}
```

### Conseils d'Optimisation

#### Performance
- **Images** : Utilisez des images nettes et bien éclairées
- **Résolution** : 300 DPI minimum pour les documents scannés
- **Format** : PNG ou JPG pour les images, PDF pour les documents

#### Instructions Efficaces
- **Soyez précis** : Listez exactement les champs souhaités
- **Format JSON** : Demandez explicitement le format JSON
- **Langue** : Utilisez la langue du document pour de meilleurs résultats