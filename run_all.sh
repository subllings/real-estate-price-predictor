#!/bin/bash

# Make this file executable: chmod +x run_all.sh
# Run it with: ./run_all.sh

clear

# === Color definitions ===
BLUE_BG="\033[44m"
GREEN_BG="\033[42m"
RED_BG="\033[41m"
WHITE_TEXT="\033[97m"
RESET="\033[0m"

# === Print helpers ===
print_blue() {
    echo ""
    echo -e "${BLUE_BG}${WHITE_TEXT}>>> $1${RESET}"
    echo ""
}

print_green() {
    echo ""
    echo -e "${GREEN_BG}${WHITE_TEXT}>>> $1${RESET}"
    echo ""
}

print_error() {
    echo ""
    echo -e "${RED_BG}${WHITE_TEXT}>>> ERROR: $1${RESET}"
    echo ""
    exit 1
}

# === Activate virtual environment (auto-detect OS) ===
print_blue "Activating virtual environment..."
if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    print_error "Could not find virtual environment activation script."
fi

# === Launch FastAPI backend ===
print_blue "Launching FastAPI backend (http://localhost:8000/docs)..."
uvicorn app.backend.main:app --reload --log-level debug &
FASTAPI_PID=$!

sleep 2

# === Launch Streamlit frontend ===
print_blue "Launching Streamlit frontend (http://localhost:8501)..."
streamlit run app/frontend-streamlit/streamlit_app.py &
STREAMLIT_PID=$!

sleep 5

# === Open browser windows ===
print_blue "Opening browser tabs..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    explorer.exe "http://localhost:8000/docs"
    explorer.exe "http://localhost:8501"
else
    xdg-open http://localhost:8000/docs >/dev/null 2>&1 &
    xdg-open http://localhost:8501 >/dev/null 2>&1 &
fi

# === Done ===
print_green "All services started successfully!"
echo "FastAPI PID     : $FASTAPI_PID"
echo "Streamlit PID   : $STREAMLIT_PID"
echo ""
echo "To stop them manually, run:"
echo "   kill $FASTAPI_PID $STREAMLIT_PID"
wait $FASTAPI_PID $STREAMLIT_PID
