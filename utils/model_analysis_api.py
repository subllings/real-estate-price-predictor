"""
API endpoints pour servir les données d'analyse des modèles à React
À ajouter dans votre FastAPI backend
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import os
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from utils.train_test_metrics_logger import TrainTestMetricsLogger
from utils.azure_model_storage import AzureModelStorage

router = APIRouter(prefix="/models", tags=["Model Analysis"])

def analyze_generalization(r2_train: float, r2_test: float, rmse_train: float, rmse_test: float) -> Dict[str, Any]:
    """Analyser la généralisation d'un modèle"""
    r2_gap = r2_train - r2_test
    
    if r2_test < 0.5:
        return {
            "category": "Underfitting",
            "color": "#ff6b6b",
            "interpretation": "Modèle trop simple, performances insuffisantes",
            "recommendation": "Augmenter la complexité, plus de features, hyperparameters"
        }
    elif r2_gap > 0.15:
        return {
            "category": "Strong overfitting", 
            "color": "#ff9f43",
            "interpretation": "Modèle mémorise les données d'entraînement",
            "recommendation": "Réduire complexité, régularisation, plus de données"
        }
    elif r2_gap > 0.08:
        return {
            "category": "Moderate overfitting",
            "color": "#feca57",
            "interpretation": "Léger surapprentissage, acceptable", 
            "recommendation": "Surveiller, possible régularisation légère"
        }
    elif r2_gap < 0.02 and r2_test > 0.7:
        return {
            "category": "Good generalization",
            "color": "#48dbfb",
            "interpretation": "Excellent équilibre train/test",
            "recommendation": "Modèle optimal, prêt pour production"
        }
    elif r2_test > 0.6 and r2_gap < 0.05:
        return {
            "category": "Light overfitting",
            "color": "#0be881", 
            "interpretation": "Bon modèle avec généralisation correcte",
            "recommendation": "Acceptable pour production, surveiller"
        }
    else:
        return {
            "category": "Moderate underfitting",
            "color": "#a55eea",
            "interpretation": "Performances moyennes, marge d'amélioration", 
            "recommendation": "Optimiser features et hyperparameters"
        }

@router.get("/analysis")
async def get_models_analysis():
    """
    Récupérer l'analyse complète de tous les modèles entraînés
    """
    try:
        # Charger les données depuis le CSV logger
        logger = TrainTestMetricsLogger()
        
        # Lire le fichier CSV directement
        csv_path = "data/model_train_test_logs/train_test_metrics.csv"
        if not os.path.exists(csv_path):
            raise HTTPException(status_code=404, detail="Fichier de métriques introuvable")
        
        df = pd.read_csv(csv_path)
        
        if len(df) == 0:
            return {
                "total_models": 0,
                "best_r2": 0,
                "mean_r2": 0,
                "best_model": "Aucun",
                "production_ready_count": 0,
                "models_summary": []
            }
        
        # Analyser chaque modèle
        models_analysis = []
        for _, row in df.iterrows():
            # Calculer les gaps
            r2_gap = row['r2_train'] - row['r2_test']
            rmse_gap = row['rmse_test'] - row['rmse_train']
            
            # Analyser la généralisation
            analysis = analyze_generalization(
                row['r2_train'], row['r2_test'], 
                row['rmse_train'], row['rmse_test']
            )
            
            model_data = {
                **row.to_dict(),
                "r2_gap": r2_gap,
                "rmse_gap": rmse_gap,
                **analysis
            }
            
            models_analysis.append(model_data)
        
        # Calculer les métriques globales
        best_r2 = df['r2_test'].max()
        mean_r2 = df['r2_test'].mean()
        best_model = df.loc[df['r2_test'].idxmax(), 'model']
        
        # Compter les modèles production-ready
        production_ready = len([
            m for m in models_analysis 
            if m['r2_test'] >= 0.7 and m['r2_gap'] <= 0.1 and 
            m['category'] not in ['Strong overfitting', 'Underfitting']
        ])
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_models": len(df),
            "best_r2": float(best_r2),
            "mean_r2": float(mean_r2),
            "best_model": best_model,
            "production_ready_count": production_ready,
            "models_summary": models_analysis
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur analyse modèles: {str(e)}")

@router.get("/categories")
async def get_model_categories():
    """
    Récupérer la distribution des catégories de modèles
    """
    try:
        analysis = await get_models_analysis()
        
        # Compter par catégorie
        categories = {}
        for model in analysis["models_summary"]:
            cat = model["category"]
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            "categories": categories,
            "total": analysis["total_models"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur catégories: {str(e)}")

@router.get("/performance-evolution")
async def get_performance_evolution():
    """
    Récupérer l'évolution des performances dans le temps
    """
    try:
        analysis = await get_models_analysis()
        
        # Trier par timestamp et formater pour graphique
        models_sorted = sorted(
            analysis["models_summary"], 
            key=lambda x: x["timestamp"]
        )
        
        evolution = [
            {
                "date": model["timestamp"],
                "r2_test": model["r2_test"],
                "r2_train": model["r2_train"],
                "rmse_test": model["rmse_test"],
                "model_name": model["model"]
            }
            for model in models_sorted
        ]
        
        return {
            "evolution": evolution,
            "trend": "improving" if len(evolution) >= 2 and evolution[-1]["r2_test"] > evolution[0]["r2_test"] else "stable"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur évolution: {str(e)}")

@router.get("/best-models/{limit}")
async def get_best_models(limit: int = 10):
    """
    Récupérer les N meilleurs modèles
    """
    try:
        analysis = await get_models_analysis()
        
        # Trier par R² décroissant
        best_models = sorted(
            analysis["models_summary"],
            key=lambda x: x["r2_test"],
            reverse=True
        )[:limit]
        
        return {
            "best_models": best_models,
            "limit": limit,
            "total_models": analysis["total_models"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur meilleurs modèles: {str(e)}")

@router.get("/production-candidates")
async def get_production_candidates():
    """
    Récupérer les modèles candidats pour la production
    """
    try:
        analysis = await get_models_analysis()
        
        # Filtrer les modèles production-ready
        candidates = [
            model for model in analysis["models_summary"]
            if (model["r2_test"] >= 0.7 and 
                model["r2_gap"] <= 0.1 and
                model["category"] in ["Good generalization", "Light overfitting"])
        ]
        
        # Trier par R² décroissant
        candidates.sort(key=lambda x: x["r2_test"], reverse=True)
        
        return {
            "candidates": candidates,
            "count": len(candidates),
            "recommendation": candidates[0] if candidates else None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur candidats production: {str(e)}")

@router.get("/azure-models")
async def get_azure_models():
    """
    Récupérer les modèles disponibles sur Azure
    """
    try:
        storage = AzureModelStorage()
        azure_models = storage.list_all_models()
        
        return {
            "azure_models": azure_models,
            "count": len(azure_models),
            "best_azure_model": azure_models[0] if azure_models else None
        }
        
    except Exception as e:
        # Si Azure n'est pas configuré, retourner une réponse vide
        return {
            "azure_models": [],
            "count": 0,
            "best_azure_model": None,
            "error": "Azure storage non configuré"
        }

@router.get("/dashboard-export")
async def export_dashboard_data():
    """
    Exporter toutes les données pour le dashboard React
    """
    try:
        # Récupérer toutes les données
        analysis = await get_models_analysis()
        categories = await get_model_categories()
        evolution = await get_performance_evolution()
        best_models = await get_best_models(10)
        production_candidates = await get_production_candidates()
        
        # Azure models (optionnel)
        try:
            azure_models = await get_azure_models()
        except:
            azure_models = {"azure_models": [], "count": 0}
        
        dashboard_data = {
            "timestamp": datetime.now().isoformat(),
            "analysis": analysis,
            "categories": categories,
            "evolution": evolution,
            "best_models": best_models,
            "production_candidates": production_candidates,
            "azure_models": azure_models
        }
        
        # Sauvegarder dans un fichier pour React
        os.makedirs("reports", exist_ok=True)
        with open("reports/dashboard_summary.json", "w") as f:
            json.dump(dashboard_data, f, indent=2, default=str)
        
        return dashboard_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur export dashboard: {str(e)}")

@router.get("/dashboard-file")
async def get_dashboard_file():
    """
    Servir le fichier JSON du dashboard pour React
    """
    file_path = "reports/dashboard_summary.json"
    if not os.path.exists(file_path):
        # Générer le fichier s'il n'existe pas
        await export_dashboard_data()
    
    return FileResponse(
        file_path,
        media_type="application/json",
        filename="dashboard_summary.json"
    )

# Fonction d'initialisation pour ajouter les routes à votre FastAPI
def add_model_analysis_routes(app):
    """
    Ajouter les routes d'analyse des modèles à votre FastAPI
    
    Usage dans votre main.py:
    from utils.model_analysis_api import add_model_analysis_routes
    add_model_analysis_routes(app)
    """
    app.include_router(router)
    print("[✅] Routes d'analyse des modèles ajoutées à FastAPI")
