#!/bin/bash

# cd e:/_SoftEng/_BeCode/real-estate-price-predictor/app/backend-api-llm-v2
# chmod +x setup-conda-env-api-llm.sh
# ./setup-conda-env-api-llm.sh

echo "🚀 Installing packages in base environment for Azure OpenAI LLM API..."

# Install FastAPI if not already installed
conda install -c conda-forge fastapi uvicorn -y

# Install other packages via pip including Azure dependencies
pip install openai langchain langchain-openai langchain-community faiss-cpu PyMuPDF python-docx python-dotenv requests pydantic azure-search-documents azure-identity azure-cosmos azure-core

# Test
python -c "import fastapi, langchain, openai; print('✅ Setup OK - using base environment')"

echo "✅ Done! All packages installed in base environment"
echo "✅ Run: ./start-api-llm-v2.sh"