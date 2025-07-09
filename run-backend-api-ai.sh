#!/bin/bash

# Make this file executable: chmod +x run-backend-api-ai.sh
# Run it with: ./run-backend-api-ai.sh

clear

# Set default host and port
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-5050}

echo -e "\e[34mStarting Node.js API at http://${HOST}:${PORT} ...\e[0m"

# Navigate to the backend directory
cd app/backend-ai

# Start the Node.js server in dev mode using nodemon with the correct port
PORT=$PORT npx nodemon index.js &

# Wait for the server to start
sleep 2

echo -e "\e[34mOpening browser...\e[0m"

# Open default browser depending on OS
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    explorer.exe "http://localhost:${PORT}/"
else
    xdg-open "http://localhost:${PORT}/" >/dev/null 2>&1 &
fi
