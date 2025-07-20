"""
API Backend pour le système de training ML
Gère les training jobs, les modèles et les hyperparamètres
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from pathlib import Path
import sys

# Ajouter le répertoire parent au path pour importer les modules
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Importer le module training_jobs depuis le dossier backend/api
try:
    from backend.api.training_jobs import router as training_jobs_router
except ImportError:
    # Si l'import échoue, créer un router vide pour éviter les erreurs
    from fastapi import APIRouter
    training_jobs_router = APIRouter()
    print("⚠️ Module training_jobs non trouvé, router vide créé")

# Configuration
app = FastAPI(
    title="Real Estate ML Training API",
    description="API pour gérer les entraînements de modèles ML immobiliers",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifier les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inclure les routers
app.include_router(training_jobs_router)

# Endpoints de base
@app.get("/")
async def root():
    """Endpoint racine de l'API"""
    return {
        "message": "Real Estate ML Training API",
        "version": "1.0.0",
        "status": "active",
        "docs": "/docs",
        "endpoints": {
            "training_jobs": "/api/training-jobs",
            "health": "/api/training-jobs/health"
        }
    }

@app.get("/health")
async def health_check():
    """Vérification de santé générale de l'API"""
    return {
        "status": "healthy",
        "service": "training-api",
        "version": "1.0.0"
    }

# Gestionnaire d'erreurs global
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Gestionnaire d'erreurs global"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Erreur interne du serveur",
            "detail": str(exc),
            "type": type(exc).__name__
        }
    )

if __name__ == "__main__":
    # Configuration du serveur
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8003))  # Port différent des autres APIs
    
    print(f"🚀 Démarrage de l'API Training sur {host}:{port}")
    print(f"📖 Documentation: http://{host}:{port}/docs")
    print(f"🔄 Health Check: http://{host}:{port}/health")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,  # Rechargement automatique en développement
        reload_dirs=[".", "../backend/api"],
        log_level="info"
    )
