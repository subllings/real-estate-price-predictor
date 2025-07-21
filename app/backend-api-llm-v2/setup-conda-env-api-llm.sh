#!/bin/bash

# cd e:/_SoftEng/_BeCode/real-estate-price-predictor/app/backend-api-llm-v2
# chmod +x setup-conda-env-api-llm.sh
# ./setup-conda-env-api-llm.sh

echo "🚀 Installing packages in base environment for Azure OpenAI LLM API..."

# Install FastAPI if not already installed
conda install -c conda-forge fastapi uvicorn -y

# Install other packages via pip
pip install openai langchain langchain-openai langchain-community faiss-cpu PyMuPDF python-docx python-dotenv requests pydantic

# Test
python -c "import fastapi, langchain, openai; print('✅ Setup OK - using base environment')"

echo "✅ Done! All packages installed in base environment"
echo "✅ Run: ./start-api-llm-v2.sh"

# Remove old environment
conda env remove -n "${ENV_NAME}" -y 2>/dev/null || true

# Create new environment
conda create -n "${ENV_NAME}" python="$PYTHON_VERSION" -y

# Activate environment
conda activate "${ENV_NAME}"

# Install packages
conda install -c conda-forge fastapi uvicorn -y
pip install openai langchain langchain-openai langchain-community faiss-cpu numpy PyMuPDF python-docx python-dotenv requests pydantic

# Test
python -c "import fastapi, langchain, openai; print('✅ Setup OK')"

echo "✅ Done! Activate with: conda activate ${ENV_NAME}"
echo "✅ Done! Run: ./start-api-llm-v2.sh"


