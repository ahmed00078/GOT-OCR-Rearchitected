#!/bin/bash
# Model Configuration and Quick Test Scripts
# Environmental Data Extraction Model Evaluation

# ==============================================
# INSTALLATION REQUIREMENTS
# ==============================================

echo "📋 INSTALLATION REQUIREMENTS"
echo "================================"

# Python dependencies
echo "🐍 Python packages needed:"
echo "pip install requests transformers torch accelerate"
echo "pip install openai  # For OpenAI models"
echo ""

# Ollama setup (if using local models)
echo "🦙 Ollama setup (for local models):"
echo "curl -fsSL https://ollama.ai/install.sh | sh"
echo "ollama serve  # Start in background"
echo ""

# ==============================================
# QUICK TEST COMMANDS
# ==============================================

echo "⚡ QUICK TEST COMMANDS"
echo "======================"

# Pattern-based baseline (always works)
echo "🔧 Test pattern-based baseline:"
echo "python model_evaluator.py --model pattern-based --quick"
echo ""

# Ollama models (if available)
echo "🦙 Test Ollama models:"
echo "python model_evaluator.py --model ollama:llama3.1 --quick"
echo "python model_evaluator.py --model ollama:phi3.5 --quick"
echo "python model_evaluator.py --model ollama:qwen2.5:3b --quick"
echo "python model_evaluator.py --model ollama:gemma2:2b --quick"
echo ""

# Transformers models (need GPU/good CPU)
echo "🤗 Test Transformers models:"
echo "python model_evaluator.py --model transformers:microsoft/Phi-3.5-mini-instruct --quick"
echo "python model_evaluator.py --model transformers:Qwen/Qwen2.5-3B-Instruct --quick"
echo "python model_evaluator.py --model transformers:google/gemma-2-2b-it --quick"
echo "python model_evaluator.py --model transformers:HuggingFaceTB/SmolLM2-1.7B-Instruct --quick"
echo ""

# OpenAI models (need API key)
echo "🔑 Test OpenAI models (set OPENAI_API_KEY):"
echo "export OPENAI_API_KEY='your_key_here'"
echo "python model_evaluator.py --model openai:gpt-3.5-turbo --quick"
echo "python model_evaluator.py --model openai:gpt-4o-mini --quick"
echo ""

# ==============================================
# COMPREHENSIVE EVALUATION COMMANDS  
# ==============================================

echo "🧪 COMPREHENSIVE EVALUATION"
echo "============================"

# Category-specific tests
echo "📊 Test by category:"
echo "python model_evaluator.py --model ollama:llama3.1 --category carbon_footprint"
echo "python model_evaluator.py --model ollama:phi3.5 --category technical_specs"
echo "python model_evaluator.py --model transformers:microsoft/Phi-3.5-mini-instruct --category complex_analysis"
echo ""

# Difficulty-specific tests
echo "🎚️ Test by difficulty:"
echo "python model_evaluator.py --model ollama:qwen2.5:3b --difficulty easy"
echo "python model_evaluator.py --model transformers:Qwen/Qwen2.5-3B-Instruct --difficulty expert"
echo ""

# Full evaluation
echo "🔬 Full evaluation (all documents):"
echo "python model_evaluator.py --model ollama:llama3.1"
echo "python model_evaluator.py --model transformers:microsoft/Phi-3.5-mini-instruct"
echo ""

# ==============================================
# COMPARISON AND ANALYSIS
# ==============================================

echo "📈 COMPARISON AND ANALYSIS"
echo "=========================="

# Compare all results
echo "🏆 Compare all existing results:"
echo "python model_evaluator.py --compare-models"
echo ""

# Compare specific models
echo "🤔 Compare specific results:"
echo "python model_evaluator.py --compare-files results/ollama_llama3.1_*.json results/transformers_microsoft_Phi-3.5_*.json"
echo ""

# ==============================================
# RECOMMENDED TESTING WORKFLOW
# ==============================================

echo "🎯 RECOMMENDED WORKFLOW"
echo "======================="
echo ""

echo "1️⃣ Start with baseline:"
echo "python model_evaluator.py --model pattern-based --quick"
echo ""

echo "2️⃣ Test your current model (SmolLM2):"
echo "python model_evaluator.py --model transformers:HuggingFaceTB/SmolLM2-1.7B-Instruct --quick"
echo ""

echo "3️⃣ Test promising alternatives:"
echo "python model_evaluator.py --model ollama:phi3.5 --quick"
echo "python model_evaluator.py --model ollama:qwen2.5:3b --quick"
echo ""

echo "4️⃣ Full evaluation of top candidates:"
echo "python model_evaluator.py --model ollama:phi3.5 --category carbon_footprint"
echo "python model_evaluator.py --model transformers:microsoft/Phi-3.5-mini-instruct --category carbon_footprint"
echo ""

echo "5️⃣ Compare results:"
echo "python model_evaluator.py --compare-models"
echo ""

# ==============================================
# MODEL RECOMMENDATIONS BY USE CASE
# ==============================================

echo "💡 MODEL RECOMMENDATIONS"
echo "========================"
echo ""

echo "🚀 For Development/Testing (Fast):"
echo "- pattern-based (instant)"
echo "- ollama:phi3.5 (2-5s per question)"
echo "- HuggingFaceTB/SmolLM2-1.7B-Instruct (1-3s)"
echo ""

echo "🎯 For Production (Accuracy):"
echo "- microsoft/Phi-3.5-mini-instruct"
echo "- Qwen/Qwen2.5-3B-Instruct"
echo "- openai:gpt-3.5-turbo (if budget allows)"
echo ""

echo "⚖️ For Balance (Speed + Accuracy):"
echo "- ollama:qwen2.5:3b"
echo "- ollama:phi3.5"
echo "- google/gemma-2-2b-it"
echo ""

echo "💻 For Limited Resources:"
echo "- pattern-based"
echo "- HuggingFaceTB/SmolLM2-1.7B-Instruct"
echo "- ollama:phi3.5 (quantized)"
echo ""

# ==============================================
# TROUBLESHOOTING
# ==============================================

echo "🔧 TROUBLESHOOTING"
echo "=================="
echo ""

echo "❌ Ollama connection failed:"
echo "  - Check ollama is running: ollama list"
echo "  - Start ollama: ollama serve"
echo "  - Pull model: ollama pull llama3.1"
echo ""

echo "❌ Transformers out of memory:"
echo "  - Use smaller models (2B instead of 7B)"
echo "  - Add --quick flag for fewer tests"
echo "  - Use Ollama instead of Transformers"
echo ""

echo "❌ OpenAI API key error:"
echo "  - Set environment: export OPENAI_API_KEY='sk-...'"
echo "  - Check key validity at https://platform.openai.com"
echo ""

echo "❌ Dataset not found:"
echo "  - Create dataset.json file"
echo "  - Or use embedded dataset (automatic fallback)"
echo ""

# ==============================================
# EXAMPLE BATCH TESTING SCRIPT
# ==============================================

echo "🔄 BATCH TESTING SCRIPT"
echo "======================="

cat << 'EOF'
#!/bin/bash
# batch_test_models.sh - Test multiple models automatically

echo "🚀 Starting batch model evaluation..."

# List of models to test
MODELS=(
    "pattern-based"
    "ollama:qwen3:8b"
    "ollama:qwen2.5:3b"
    "transformers:HuggingFaceTB/SmolLM2-1.7B-Instruct"
)

# Test each model
for model in "${MODELS[@]}"; do
    echo "Testing $model..."
    python model_evaluator.py --model "$model" --quick
    echo "---"
done

# Compare results
echo "Comparing all results..."
python model_evaluator.py --compare-models

echo "✅ Batch testing complete!"
EOF

echo ""
echo "💾 Save the above as 'batch_test_models.sh' and run with: bash batch_test_models.sh"
echo ""

# ==============================================
# DATASET CUSTOMIZATION
# ==============================================

echo "📝 DATASET CUSTOMIZATION"
echo "========================"
echo ""

echo "🔧 To add your own test cases:"
echo "1. Edit dataset.json"
echo "2. Add new document with your real OCR text"
echo "3. Add questions and expected answers"
echo "4. Test: python model_evaluator.py --model your_model --max-docs 1"
echo ""

echo "💡 Document structure:"
cat << 'EOF'
{
  "id": 99,
  "category": "carbon_footprint",
  "difficulty": "medium",
  "title": "Your Real Document",
  "text": "Your actual OCR text here...",
  "questions": [
    {
      "id": "q99_1",
      "question": "Extract the CO2 emissions",
      "expected_answer": {
        "emissions": {"value": 123, "unit": "kg CO2"}
      }
    }
  ]
}
EOF

echo ""
echo "🎉 You're ready to test! Start with:"
echo "python model_evaluator.py --model pattern-based --quick"