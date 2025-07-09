#!/bin/bash

# Make this file executable: chmod +x run-frontend-react.sh
# Run it with: ./run-frontend-react.sh

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

# === Navigate into frontend-react directory ===
print_blue "Navigating to 'app/frontend-react' directory..."
cd "$(dirname "$0")/app/frontend-react" || print_error "'app/frontend-react' directory not found."

# === Check for node_modules and react-scripts ===
if [ ! -d "node_modules" ] || [ ! -f "node_modules/.bin/react-scripts" ]; then
    print_blue "Installing npm dependencies including react-scripts..."
    rm -rf node_modules package-lock.json
    npm install || print_error "npm install failed."
else
    print_green "Dependencies already installed (node_modules and react-scripts found)."
fi

# === Start React app ===
print_blue "Launching React application..."

# Run react-scripts directly from local node_modules to avoid global path issues
./node_modules/.bin/react-scripts start || print_error "react-scripts start failed."

# === Done ===
print_green "React application is running. Press Ctrl+C to stop."
