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

<<<<<<< HEAD
# Force use environment python directly
PYTHON_EXE="${ENV_PATH}/python"
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    PYTHON_EXE="${ENV_PATH}/python.exe"
fi

# Install everything directly with the environment python
echo ">>> Installing all packages directly..."
"$PYTHON_EXE" -m pip install --upgrade pip --no-warn-script-location
"$PYTHON_EXE" -m pip install fastapi uvicorn python-multipart openai langchain langchain-openai langchain-community faiss-cpu numpy PyMuPDF python-docx python-dotenv requests pydantic azure-search-documents azure-identity azure-cosmos azure-core --no-warn-script-location

# Test
"$PYTHON_EXE" -c "import fastapi, langchain, openai; print('✅ Setup OK')"
=======
# Install packages
conda install -c conda-forge fastapi uvicorn -y
pip install openai langchain langchain-openai langchain-community faiss-cpu numpy PyMuPDF python-docx python-dotenv requests pydantic

# Test
python -c "import fastapi, langchain, openai; print('✅ Setup OK')"
>>>>>>> 915073c6775d41e055b58716734b8e21e488a33f

echo "✅ Done! Activate with: conda activate ${ENV_NAME}"
echo "✅ Done! Run: ./start-api-llm-v2.sh"
<<<<<<< HEAD

# Force use environment python directly
PYTHON_EXE="${ENV_PATH}/python"
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    PYTHON_EXE="${ENV_PATH}/python.exe"
fi

# Install everything directly with the environment python
echo -e "\033[33m>>> Installing all packages directly...\033[0m"
"$PYTHON_EXE" -m pip install --upgrade pip
"$PYTHON_EXE" -m pip install fastapi uvicorn python-multipart openai langchain langchain-openai langchain-community faiss-cpu numpy PyMuPDF python-docx python-dotenv requests pydantic

# Test
"$PYTHON_EXE" -c "import fastapi, langchain, openai; print('✅ Setup OK')"

echo -e "\033[32m✅ Setup terminé malgré les warnings!\033[0m"
echo ""
echo -e "\033[34m📋 Next steps:\033[0m"
echo -e "\033[32m  1. Run the API: ./start-api-llm-v2.sh\033[0m"
echo -e "\033[32m  2. Configure .env file with Azure OpenAI settings\033[0m"
echo ""
echo -e "\033[33m💡 Environment info:\033[0m"
echo -e "\033[33m   Path: $ENV_PATH\033[0m"
echo -e "\033[33m   Python: $PYTHON_VERSION\033[0m"
echo -e "\033[33m   Purpose: Azure OpenAI LLM API\033[0m"

# Utilities
echo -e "\033[33m>>> Installing utility libraries...\033[0m"
python -m pip install python-dotenv==1.0.0
python -m pip install requests==2.31.0
python -m pip install pydantic>=2.7.4

# Verify installation
echo -e "\033[33m>>> Verifying installation...\033[0m"
python -c "
try:
    import fastapi, uvicorn, langchain, openai, fitz, docx, faiss
    print('✅ All core dependencies installed successfully')
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
    exit(1)
"

# Create requirements.txt for future reference
echo -e "\033[33m>>> Creating requirements.txt...\033[0m"
pip freeze > requirements.txt

echo -e "\033[32m✅ Local environment '$ENV_PATH' created successfully!\033[0m"
echo ""
echo -e "\033[34m📋 Next steps:\033[0m"
echo -e "\033[32m  1. Activate environment: conda activate $ENV_PATH\033[0m"
echo -e "\033[32m  2. Configure .env file with Azure OpenAI settings\033[0m"
echo -e "\033[32m  3. Run the API: ./start-api.sh\033[0m"
echo ""
echo -e "\033[33m💡 Environment info:\033[0m"
echo -e "\033[33m   Path: $ENV_PATH\033[0m"
echo -e "\033[33m   Python: $PYTHON_VERSION\033[0m"
echo -e "\033[33m   Location: $(realpath $ENV_PATH 2>/dev/null || echo $ENV_PATH)\033[0m"
=======
>>>>>>> 915073c6775d41e055b58716734b8e21e488a33f

# Install additional packages via pip
pip install fastapi uvicorn python-multipart python-dotenv requests pydantic langchain langchain-community langchain-openai azure-search-documents azure-identity pymupdf python-docx azure-cosmos azure-core --no-warn-script-location


