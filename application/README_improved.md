# 🚀 Pipeline PDF Améliorée

Version optimisée avec validation JSON stricte, logique de retry et prompts spécialisés par type de document.

## 🎯 Améliorations implémentées

### 1. **Validation JSON stricte**
- Filtrage automatique des champs parasites
- Extraction de JSON depuis réponses malformées
- Validation des types de données (numérique vs texte)
- Nettoyage automatique des unités

### 2. **Logique de retry intelligente**
- Nouvelle tentative si extraction échoue ou < 2 champs trouvés
- Prompt plus strict au 2ème essai
- Statistiques des tentatives multiples

### 3. **Détection automatique du type de document**
- **Apple**: EcoProfile, MacBook, iMac
- **HP**: Product Carbon Footprint, PAIA tool
- **Lenovo**: PCF-, IdeaPad, ThinkPad  
- **Acer**: Veriton, PAIA algorithm
- **Microsoft**: Surface, Xbox
- **Generic**: Fallback pour autres documents

### 4. **Prompts spécialisés par marque**
- Mots-clés spécifiques par fabricant
- Instructions adaptées au format du document
- Contraintes JSON ultra-strictes

## 📊 Résultats attendus

Avec **Qwen2.5-7B-Instruct** (vs SmolLM2):

| Métrique | Avant | Après (estimé) |
|----------|--------|---------|
| **Taux de succès JSON** | 67.7% | **90%+** |
| **Champs parasites** | 41.9% | **<5%** |
| **Extraction manufacturer** | 64.5% | **85%+** |
| **Extraction carbon_footprint** | 29.0% | **60%+** |
| **Réponses malformées** | 32.3% | **<10%** |

## 🛠️ Utilisation

### Test rapide
```bash
cd application/
source ../venv/bin/activate
python test_improved_pipeline.py
```

### Utilisation complète
```python
from improved_pipeline import ImprovedPDFPipeline

async def run_improved():
    pipeline = ImprovedPDFPipeline()
    await pipeline.initialize()
    
    results = await pipeline.run_batch_processing(
        pdf_folder="data",
        output_file="improved_results.json"
    )
    
    # Comparer avec anciens résultats
    stats = pipeline.get_improvement_stats("equipment_extraction_results.json")
    print(f"Amélioration: {stats['improvement']:+.1f}%")

asyncio.run(run_improved())
```

## 📋 Fonctionnalités avancées

### Détection automatique de type
```python
from improved_pipeline import DocumentTypeDetector

detector = DocumentTypeDetector()
doc_type = detector.detect_type(text, filename)
# Retourne: 'apple', 'hp', 'lenovo', 'acer', 'microsoft', 'generic'
```

### Validation JSON robuste
```python
from improved_pipeline import JSONValidator

# Nettoie automatiquement les réponses
cleaned = JSONValidator.validate_and_clean(raw_response)

# Extrait JSON depuis texte malformé
json_data = JSONValidator._extract_json_from_text(malformed_text)
```

### Prompts optimisés
```python
from improved_pipeline import ImprovedPromptGenerator

generator = ImprovedPromptGenerator()

# Prompt spécialisé Acer
acer_prompt = generator.generate_strict_prompt('acer')

# Prompt générique strict  
strict_prompt = generator.generate_strict_prompt('generic')
```

## 🔍 Exemple de prompt optimisé

```
CRITICAL INSTRUCTIONS - FOLLOW EXACTLY:

Look for: Acer brand, Product Carbon Footprint, PAIA algorithm, typical energy consumption

Extract ONLY these 6 fields in this EXACT JSON format:

{
  "manufacturer": "string or null",
  "year": "string or null",
  "product_name": "string or null", 
  "carbon_footprint": "number or null",
  "power_consumption": "number or null",
  "weight": "number or null"
}

CRITICAL RULES:
1. Response must be VALID JSON only - no markdown, no explanations
2. Use EXACTLY these 6 field names - no additional fields allowed
3. Use null for missing information
4. Extract only numeric values for carbon_footprint, power_consumption, weight
5. Do not include units in numeric fields
6. Do not add any text before or after the JSON

JSON Response:
```

## 🧪 Tests et validation

### Tests inclus
1. **Test complet**: Compare avec résultats précédents
2. **Test fichier problématique**: Vérifie la correction des erreurs JSON
3. **Statistiques d'amélioration**: Mesure les gains par champ
4. **Analyse par type de document**: Performance par fabricant

### Métriques suivies
- Taux de succès global
- Nombre de retries nécessaires
- Performance par type de document
- Amélioration par champ extrait
- Temps de traitement moyen

## 🚨 Avec Qwen2.5-7B vs SmolLM2

### Avantages de Qwen2.5-7B:
✅ **Meilleur suivi d'instructions JSON strictes**  
✅ **Contexte plus long** (32K tokens vs 4K)  
✅ **Moins d'hallucinations** et champs inventés  
✅ **Parsing JSON plus fiable**  
✅ **Compréhension multilingue** améliorée  

### Impact attendu:
- **90%+ de réponses JSON valides** (vs 67.7%)
- **Réduction drastique des champs parasites**
- **Extraction plus précise des valeurs numériques**
- **Moins de retries nécessaires**

## 🔧 Configuration recommandée

Pour tester avec Qwen2.5-7B, modifiez dans votre config:
```python
# config.py
REASONING_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_TOKENS = 512  # Suffisant pour JSON strict
TEMPERATURE = 0.1     # Très bas pour consistance
```

La pipeline améliorée devrait résoudre **90%+ des problèmes identifiés** et atteindre un taux de succès de **85-90%** avec le nouveau modèle.