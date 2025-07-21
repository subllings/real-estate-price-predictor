#!/bin/bash

# cd e:/_SoftEng/_BeCode/real-estate-price-predictor/app/frontend-react
# chmod +x start-react-clean.sh
# ./start-react-clean.sh

echo "🚀 Starting React app with clean logs..."

# Start React with minimal logs
REACT_APP_LOG_LEVEL=error npm start 2>&1 | grep -v "webpack\|DevTools\|compiled"
