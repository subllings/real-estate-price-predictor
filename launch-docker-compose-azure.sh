#!/bin/bash

# Make this file executable: chmod +x launch-docker-compose-azure.sh
# Run it with: ./launch-docker-compose-azure.sh

clear

# === Define color codes ===
BLUE_BG="\033[44m"
GREEN_BG="\033[42m"
RED_BG="\033[41m"
WHITE_TEXT="\033[97m"
RESET="\033[0m"

# === Define print helpers ===
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

docker ps -a --filter "ancestor=test-streamlit-ui"
ddocker rm -f $(docker ps -aq --filter "ancestor=test-streamlit-ui")

# === Build Docker image ===
print_blue "Building Docker image..."
docker build -f app/frontend-streamlit/Dockerfile.azure -t test-streamlit-ui app/frontend-streamlit || print_error "Docker build failed."

# === Run Docker container ===
print_blue "Running Docker container..."
docker run -p 8501:8501 test-streamlit-ui || print_error "Docker run failed."

# === Success ===
print_green "Streamlit app running at http://localhost:8501"


