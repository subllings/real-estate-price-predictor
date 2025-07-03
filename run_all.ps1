# To run this script in PowerShell:
# 1. Right-click > "Run with PowerShell", or
# 2. In terminal, execute: .\run_all.ps1

# install_frontend_react.ps1
# PowerShell script to install React frontend dependencies only once

try {
    # Try to get the full path of node.exe
    $nodeFullPath = (Get-Command node -ErrorAction Stop).Path
    Write-Host "Node.js executable found at: $nodeFullPath"
} catch {
    Write-Host "Node.js executable not found in PATH."
    exit 1
}

# Add Node.js folder to current session PATH
$nodePath = Split-Path $nodeFullPath
$env:PATH = "$nodePath;$env:PATH"

Write-Host "Launching FastAPI backend in a new PowerShell window..."
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd '$(Get-Location)'; .\.venv\Scripts\Activate.ps1; uvicorn app.backend.main:app --reload"
)

Start-Sleep -Seconds 2

Write-Host "Launching React frontend in a new PowerShell window..."
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd app/frontend-react; npm start"
)

Start-Sleep -Seconds 5

Write-Host "Opening browser for React frontend..."
Start-Process "http://localhost:3000"

Write-Host "Opening browser for FastAPI docs..."
Start-Process "http://localhost:8000/docs"

Write-Host "All processes started successfully."
