#!/bin/bash

# Make this file executable: chmod +x launch-docker-compose-dev.sh
# Run it with: ./launch-docker-compose-dev.sh

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

# === Check Docker availability ===
print_blue "Checking Docker installation..."
docker -v || print_error "Docker is not installed or not available in PATH."

# === Check docker-compose file ===
print_blue "Looking for docker-compose.yml..."
if [ ! -f "docker-compose.yml" ]; then
    print_error "docker-compose.yml not found in current directory. Run this script from the project root."
fi

# === Launch containers ===
print_blue "Building and launching Docker containers..."
docker compose up --build || print_error "Docker Compose failed to start."


