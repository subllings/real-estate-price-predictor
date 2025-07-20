@echo off
echo Killing existing React processes...
taskkill /F /IM node.exe 2>nul
timeout /t 2 /nobreak > nul
echo Starting React application...
cd app\frontend-react
start cmd /k "npm start"
echo React application started!
