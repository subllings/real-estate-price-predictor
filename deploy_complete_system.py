#!/usr/bin/env python3
"""
Script de déploiement automatique complet
- Entraîne le meilleur modèle
- Upload vers Azure
- Met à jour FastAPI
- Génère le dashboard
"""

import sys
import os
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

def setup_azure_integration():
    """
    Configurer l'intégration Azure pour le projet
    """
    print("🔧 CONFIGURATION AZURE INTÉGRATION")
    print("=" * 50)
    
    # Vérifier les variables d'environnement
    required_vars = [
        "AZURE_STORAGE_CONNECTION_STRING",
        "AZURE_MODELS_CONTAINER"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Variables d'environnement manquantes:")
        for var in missing_vars:
            print(f"   - {var}")
        
        print("\n💡 Ajouter à votre .env:")
        print("AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...")
        print("AZURE_MODELS_CONTAINER=ml-models")
        return False
    
    print("✅ Configuration Azure OK")
    return True

def test_azure_connection():
    """
    Tester la connexion Azure
    """
    print("\n🔍 TEST CONNEXION AZURE")
    print("-" * 30)
    
    try:
        from utils.azure_model_storage import AzureModelStorage
        
        storage = AzureModelStorage()
        models = storage.list_all_models()
        
        print(f"✅ Connexion Azure réussie")
        print(f"📊 {len(models)} modèles trouvés sur Azure")
        
        if models:
            best = models[0]  # Déjà trié par R²
            print(f"🏆 Meilleur modèle: R² = {best.get('r2_test', 0):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur connexion Azure: {e}")
        return False

def deploy_best_model():
    """
    Déployer le meilleur modèle pour FastAPI
    """
    print("\n🚀 DÉPLOIEMENT MODÈLE FASTAPI")
    print("-" * 35)
    
    try:
        from utils.azure_model_storage import ensure_best_model_available
        
        api_model_path = "models/current_best_model.pkl"
        success = ensure_best_model_available(api_model_path)
        
        if success:
            print(f"✅ Modèle déployé: {api_model_path}")
            
            # Vérifier les métadonnées
            metadata_path = api_model_path.replace('.pkl', '_metadata.json')
            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                print(f"📊 R² Test: {metadata.get('r2_test', 0):.3f}")
                print(f"💰 RMSE: {metadata.get('rmse_test', 0):.0f}€")
                print(f"🔧 Features: {metadata.get('n_features', 0)}")
                
            return True
        else:
            print("❌ Échec déploiement modèle")
            return False
            
    except Exception as e:
        print(f"❌ Erreur déploiement: {e}")
        return False

def generate_dashboard_report():
    """
    Générer un rapport dashboard automatique
    """
    print("\n📊 GÉNÉRATION DASHBOARD")
    print("-" * 28)
    
    try:
        from utils.train_test_metrics_logger import TrainTestMetricsLogger
        import pandas as pd
        
        logger = TrainTestMetricsLogger()
        df = logger.get_dataframe()
        
        if len(df) == 0:
            print("⚠️ Aucune donnée trouvée dans les logs")
            return False
        
        # Analyser les performances
        best_r2 = df['r2_test'].max()
        mean_r2 = df['r2_test'].mean()
        best_model = df.loc[df['r2_test'].idxmax(), 'model']
        
        # Compter par catégorie (si analyse déjà faite)
        production_ready = len(df[(df['r2_test'] >= 0.7)])
        
        print(f"✅ {len(df)} modèles analysés")
        print(f"🏆 Meilleur R²: {best_r2:.3f} ({best_model})")
        print(f"📊 R² moyen: {mean_r2:.3f}")
        print(f"🚀 Production-ready: {production_ready} modèles")
        
        # Sauvegarder rapport JSON pour React
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_models": len(df),
            "best_r2": float(best_r2),
            "mean_r2": float(mean_r2),
            "best_model": best_model,
            "production_ready_count": int(production_ready),
            "models_summary": df.to_dict('records')
        }
        
        report_path = "reports/dashboard_summary.json"
        os.makedirs("reports", exist_ok=True)
        
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📁 Rapport sauvé: {report_path}")
        return True
        
    except Exception as e:
        print(f"❌ Erreur génération dashboard: {e}")
        return False

def check_fastapi_integration():
    """
    Vérifier si FastAPI est prêt pour l'intégration
    """
    print("\n🔌 VÉRIFICATION FASTAPI")
    print("-" * 25)
    
    # Vérifier si les dossiers existent
    required_dirs = ["models", "app"]
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            print(f"⚠️ Dossier manquant: {dir_name}")
        else:
            print(f"✅ Dossier OK: {dir_name}")
    
    # Vérifier le modèle actuel
    model_path = "models/current_best_model.pkl"
    metadata_path = "models/current_best_model_metadata.json"
    
    model_ok = os.path.exists(model_path)
    metadata_ok = os.path.exists(metadata_path)
    
    print(f"📁 Modèle actuel: {'✅' if model_ok else '❌'}")
    print(f"📋 Métadonnées: {'✅' if metadata_ok else '❌'}")
    
    if model_ok and metadata_ok:
        print("🎉 FastAPI prêt pour l'intégration!")
        print("\n💡 Ajouter à votre main.py:")
        print("from utils.fastapi_model_integration import setup_model_integration")
        print("setup_model_integration(app)")
        return True
    else:
        print("⚠️ Configuration FastAPI incomplète")
        return False

def main():
    """
    Orchestrateur principal du déploiement
    """
    print("🚀 DÉPLOIEMENT AUTOMATIQUE COMPLET")
    print("=" * 60)
    print(f"🕒 Démarrage: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    steps = [
        ("Configuration Azure", setup_azure_integration),
        ("Test connexion Azure", test_azure_connection),
        ("Déploiement modèle", deploy_best_model),
        ("Dashboard report", generate_dashboard_report),
        ("Vérification FastAPI", check_fastapi_integration)
    ]
    
    results = {}
    
    for step_name, step_func in steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        try:
            result = step_func()
            results[step_name] = result
            
            if result:
                print(f"✅ {step_name}: SUCCÈS")
            else:
                print(f"⚠️ {step_name}: PROBLÈME")
                
        except Exception as e:
            print(f"❌ {step_name}: ERREUR - {e}")
            results[step_name] = False
    
    # Résumé final
    print(f"\n🎯 RÉSUMÉ FINAL")
    print("=" * 30)
    
    success_count = sum(1 for result in results.values() if result)
    total_steps = len(steps)
    
    for step_name, result in results.items():
        status = "✅" if result else "❌"
        print(f"{status} {step_name}")
    
    print(f"\n📊 Score: {success_count}/{total_steps} étapes réussies")
    
    if success_count == total_steps:
        print("🎉 DÉPLOIEMENT COMPLET RÉUSSI!")
        print("\n🚀 PROCHAINES ÉTAPES:")
        print("   1. Redémarrer FastAPI")
        print("   2. Tester /model/info endpoint")
        print("   3. Intégrer le dashboard React")
        print("   4. Vérifier les prédictions")
    elif success_count >= 3:
        print("⚠️ DÉPLOIEMENT PARTIEL - Vérifiez les étapes échouées")
    else:
        print("❌ DÉPLOIEMENT ÉCHOUÉ - Configuration nécessaire")
    
    print(f"\n🕒 Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return success_count == total_steps

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
