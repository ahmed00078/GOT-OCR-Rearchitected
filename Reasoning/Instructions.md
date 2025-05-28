### **1. Installation des dépendances**

```bash
# Pour tester les modèles Transformers
pip install transformers torch accelerate bitsandbytes

# Pour tester GPT (optionnel)
pip install openai

# Dépendances de base
pip install numpy pandas matplotlib
```

### **2. Utilisation du script**

```bash
# Test d'un modèle spécifique sur tous les cas
python test_reasoning_models.py --model pattern-based

# Test sur un cas spécifique  
python test_reasoning_models.py --model phi-3.5 --test-id 1

# Sauvegarder le rapport
python test_reasoning_models.py --model qwen2.5 --output rapport_qwen.json
```

### **3. Exemples de résultats attendus**

```
🧪 Test du modèle: pattern-based
✅ Modèle pattern-based chargé

📄 Test 1: laptop_simple
  ❓ Question 1: What is the product name and model?...
    📊 Score: 0.85 (0.01s)
  ❓ Question 2: What are the power consumption values?...
    📊 Score: 0.95 (0.01s)

====================================================
📊 RAPPORT FINAL
====================================================
Modèle: pattern-based
Tests réussis: 18/20
Score moyen: 0.73
Temps moyen: 0.02s

📈 Scores par type de test:
  laptop_simple: 0.90
  monitor_technical: 0.78
  server_complex: 0.65
  ...
```

## 🎯 **Types de tests dans le dataset**

### **🟢 Tests simples (scores attendus élevés)**
- **Test 1** : Laptop HP (données basiques)
- **Test 6** : Tablet Samsung (unités mixtes)

### **🟡 Tests intermédiaires** 
- **Test 2** : Monitor Dell (formats variés)
- **Test 5** : Printer Canon (tableaux)
- **Test 9** : TV LG (comparaisons multiples)

### **🔴 Tests difficiles**
- **Test 3** : Server Microsoft (analyse complexe)
- **Test 4** : iPhone (OCR bruité à corriger)
- **Test 8** : Workstation Dell (lifecycle détaillé)
- **Test 10** : PlayStation (analyse complète)

### **🎲 Test baseline**
- **Test 7** : Router Netgear (données manquantes)

## 💡 **Comment interpréter les résultats**

### **Scores de référence attendus :**

| Modèle | Score moyen | Temps | Commentaire |
|--------|-------------|-------|-------------|
| Pattern-based | 0.60-0.75 | <0.1s | ⚡ Rapide mais limité |
| Phi-3.5 Mini | 0.80-0.90 | 1-3s | 🎯 Optimal pour ton cas |
| Qwen2.5-3B | 0.75-0.85 | 0.5-2s | ⚡ Bon compromis vitesse/qualité |
| GPT-3.5 | 0.85-0.92 | 0.5-1s | 💰 Excellent mais payant |

### **Critères de décision :**

- **Score > 0.8** : Excellent pour ton stage
- **Temps < 2s** : Acceptable pour usage interactif  
- **Tests difficiles > 0.7** : Robuste pour docs complexes

## 🔬 **Personnalisation pour ton domaine**

Tu peux facilement ajouter tes propres cas de test :

```python
# Ajouter dans le dataset
{
    "id": 11,
    "type": "ton_cas_specifique", 
    "text": "Contenu d'un vrai doc de constructeur...",
    "questions": [
        {
            "question": "Extraire la consommation en mode économique",
            "expected_answer": {"eco_mode": {"value": 12, "unit": "W"}}
        }
    ]
}
```

## 🎪 **Tests que tu peux faire dès maintenant**

1. **Test baseline** : Lance le pattern-based pour voir les scores de référence
2. **Test avec tes docs** : Remplace le texte par du vrai contenu OCR de ton projet
3. **Comparaison modèles** : Teste 2-3 modèles différents sur le même cas
4. **Évaluation temps** : Mesure les performances sur ton hardware