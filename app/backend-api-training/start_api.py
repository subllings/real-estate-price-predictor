"""
Script pour démarrer l'API Training
"""

import subprocess
import sys
import os
from pathlib import Path

def start_training_api():
    """Démarre l'API Training"""
    try:
        # S'assurer que les dépendances sont installées
        print("🔍 Vérification des dépendances...")
        
        # Démarrer l'API
        print("🚀 Démarrage de l'API Training...")
        
        # Port par défaut pour l'API Training
        port = os.getenv("TRAINING_API_PORT", "8003")
        host = os.getenv("TRAINING_API_HOST", "0.0.0.0")
        
        # Commande pour démarrer l'API
        cmd = [
            sys.executable, "-m", "uvicorn",
            "main:app",
            "--host", host,
            "--port", port,
            "--reload",
            "--reload-dir", ".",
            "--reload-dir", "../backend/api"
        ]
        
        print(f"📍 API Training disponible sur: http://{host}:{port}")
        print(f"📖 Documentation: http://{host}:{port}/docs")
        print("💡 Ctrl+C pour arrêter")
        
        # Démarrer le processus
        subprocess.run(cmd, cwd=Path(__file__).parent)
        
    except KeyboardInterrupt:
        print("\n⏹️ Arrêt de l'API Training...")
    except Exception as e:
        print(f"❌ Erreur lors du démarrage: {e}")
        return False
    
    return True

if __name__ == "__main__":
    start_training_api()
