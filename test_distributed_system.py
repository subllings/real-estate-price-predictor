#!/usr/bin/env python3
"""
Script de test pour le système d'entraînement distribué
Valide que tous les composants fonctionnent correctement
"""
import os
import sys
import json
import time
from pathlib import Path

def test_component(name, test_func):
    """Tester un composant et afficher le résultat"""
    print(f"🧪 Test: {name}...")
    try:
        result = test_func()
        if result:
            print(f"✅ {name}: OK")
            return True
        else:
            print(f"❌ {name}: ÉCHEC")
            return False
    except Exception as e:
        print(f"❌ {name}: ERREUR - {e}")
        return False

def test_azure_connection():
    """Tester la connexion Azure"""
    try:
        from utils.azure_model_storage import AzureModelStorage
        storage = AzureModelStorage()
        models = storage.list_all_models()
        print(f"   📊 {len(models)} modèles trouvés sur Azure")
        return True
    except Exception as e:
        print(f"   ⚠️ Azure: {e}")
        return False

def test_distributed_manager():
    """Tester le gestionnaire distribué"""
    try:
        from distributed_training_manager import DistributedTrainingManager
        manager = DistributedTrainingManager()
        print(f"   🖥️ Machine ID: {manager.machine_id}")
        print(f"   🎭 Rôle: {manager.machine_role}")
        
        # Test heartbeat
        manager.start_heartbeat()
        time.sleep(2)
        manager.stop_heartbeat()
        
        return True
    except Exception as e:
        print(f"   ⚠️ Manager: {e}")
        return False

def test_auto_recovery():
    """Tester le système de récupération automatique"""
    try:
        from auto_recovery_system import AutoRecoverySystem
        recovery = AutoRecoverySystem()
        print(f"   🔧 Système de récupération initialisé")
        return True
    except Exception as e:
        print(f"   ⚠️ Recovery: {e}")
        return False

def test_file_permissions():
    """Tester les permissions des fichiers"""
    scripts = [
        "launch_desktop_master.sh",
        "launch_laptop_slave.sh", 
        "sync_machines.sh"
    ]
    
    all_ok = True
    for script in scripts:
        if Path(script).exists():
            if os.access(script, os.X_OK):
                print(f"   ✅ {script}: exécutable")
            else:
                print(f"   ❌ {script}: pas exécutable")
                all_ok = False
        else:
            print(f"   ❌ {script}: manquant")
            all_ok = False
    
    return all_ok

def test_environment_variables():
    """Tester les variables d'environnement"""
    required_vars = [
        "AZURE_STORAGE_CONNECTION_STRING"
    ]
    
    all_ok = True
    for var in required_vars:
        if os.getenv(var):
            print(f"   ✅ {var}: configuré")
        else:
            print(f"   ❌ {var}: manquant")
            all_ok = False
    
    return all_ok

def test_python_packages():
    """Tester les packages Python requis"""
    packages = [
        "azure.storage.blob",
        "sklearn",
        "catboost",
        "pandas",
        "numpy"
    ]
    
    all_ok = True
    for package in packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"   ✅ {package}: installé")
        except ImportError:
            print(f"   ❌ {package}: manquant")
            all_ok = False
    
    return all_ok

def test_data_availability():
    """Tester la disponibilité des données"""
    data_paths = [
        "data/cleaned",
        "data/ml_ready"
    ]
    
    all_ok = True
    for path in data_paths:
        if Path(path).exists():
            files = list(Path(path).glob("*.csv"))
            print(f"   ✅ {path}: {len(files)} fichiers")
        else:
            print(f"   ❌ {path}: manquant")
            all_ok = False
    
    return all_ok

def create_test_status():
    """Créer un fichier de statut de test"""
    test_status = {
        "machines": {
            "test_machine": {
                "machine_id": "test_machine",
                "machine_role": "master",
                "last_heartbeat": "2024-01-01T00:00:00",
                "training_status": "idle",
                "progress_percent": 0,
                "current_trial": 0,
                "best_r2": 0,
                "azure_synced": True
            }
        },
        "current_master": "test_machine",
        "last_update": "2024-01-01T00:00:00"
    }
    
    with open("test_distributed_status.json", "w") as f:
        json.dump(test_status, f, indent=2)
    
    print("   📝 Fichier de test créé")
    return True

def run_full_test():
    """Lancer tous les tests"""
    print("🧪 TEST DU SYSTÈME D'ENTRAÎNEMENT DISTRIBUÉ")
    print("=" * 50)
    print()
    
    tests = [
        ("Variables d'environnement", test_environment_variables),
        ("Packages Python", test_python_packages),
        ("Connexion Azure", test_azure_connection),
        ("Permissions fichiers", test_file_permissions),
        ("Données disponibles", test_data_availability),
        ("Gestionnaire distribué", test_distributed_manager),
        ("Système de récupération", test_auto_recovery),
        ("Création statut test", create_test_status)
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_component(test_name, test_func)
        results.append((test_name, result))
        print()
    
    # Résumé
    print("📋 RÉSUMÉ DES TESTS")
    print("=" * 30)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} - {test_name}")
    
    print()
    print(f"📊 Score: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 Tous les tests sont passés!")
        print("✅ Le système est prêt pour l'entraînement distribué")
        print()
        print("🚀 ÉTAPES SUIVANTES:")
        print("   1. Desktop: ./launch_desktop_master.sh")
        print("   2. Laptop: ./launch_laptop_slave.sh")
    else:
        print("⚠️ Certains tests ont échoué")
        print("🔧 Corrigez les erreurs avant de lancer l'entraînement")
    
    # Nettoyer
    if Path("test_distributed_status.json").exists():
        os.remove("test_distributed_status.json")

if __name__ == "__main__":
    run_full_test()
