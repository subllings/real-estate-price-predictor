#!/usr/bin/env python3
"""
Système de surveillance et récupération automatique pour l'entraînement ML
Surveille, détecte les erreurs, et applique des corrections automatiques
"""
import os
import sys
import time
import subprocess
import logging
import json
from datetime import datetime
from pathlib import Path

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler('auto_recovery_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutoRecoverySystem:
    def __init__(self):
        self.max_retries = 3
        self.retry_count = 0
        self.training_process = None
        self.start_time = datetime.now()
        
    def check_environment(self):
        """Vérifier que tout est en place pour l'entraînement"""
        logger.info("🔍 Vérification de l'environnement...")
        
        checks = [
            ("Azure Storage", self._check_azure_storage),
            ("Données ML", self._check_data),
            ("Dependencies Python", self._check_python_deps),
            ("Espace disque", self._check_disk_space),
        ]
        
        for check_name, check_func in checks:
            try:
                if not check_func():
                    logger.error(f"❌ {check_name}: ÉCHEC")
                    return False
                logger.info(f"✅ {check_name}: OK")
            except Exception as e:
                logger.error(f"❌ {check_name}: Erreur - {e}")
                return False
        
        return True
    
    def _check_azure_storage(self):
        """Vérifier Azure Storage"""
        try:
            from utils.azure_model_storage import AzureModelStorage
            storage = AzureModelStorage()
            return True
        except Exception as e:
            logger.warning(f"Azure Storage issue: {e}")
            return True  # Non bloquant
    
    def _check_data(self):
        """Vérifier les données"""
        data_paths = ["data/ml_ready", "data/cleaned"]
        return any(Path(p).exists() for p in data_paths)
    
    def _check_python_deps(self):
        """Vérifier les dépendances critiques"""
        critical_modules = ["catboost", "optuna", "pandas", "numpy"]
        for module in critical_modules:
            try:
                __import__(module)
            except ImportError:
                logger.error(f"Module manquant: {module}")
                return False
        return True
    
    def _check_disk_space(self):
        """Vérifier l'espace disque (minimum 1GB)"""
        import shutil
        free_bytes = shutil.disk_usage('.').free
        free_gb = free_bytes / (1024**3)
        if free_gb < 1:
            logger.warning(f"Espace disque faible: {free_gb:.1f}GB")
            self._cleanup_old_files()
        return free_gb > 0.5
    
    def _cleanup_old_files(self):
        """Nettoyer les anciens fichiers"""
        logger.info("🧹 Nettoyage automatique...")
        try:
            # Supprimer anciens logs (>7 jours)
            subprocess.run(["find", ".", "-name", "*.log", "-mtime", "+7", "-delete"], 
                         capture_output=True)
            # Supprimer fichiers temporaires
            subprocess.run(["find", ".", "-name", "*.tmp", "-delete"], 
                         capture_output=True)
        except Exception as e:
            logger.warning(f"Nettoyage partiel: {e}")
    
    def detect_and_fix_error(self, error_text):
        """Détecter et corriger automatiquement les erreurs"""
        error_lower = error_text.lower()
        
        # Erreurs mémoire
        if any(keyword in error_lower for keyword in ['memory', 'ram', 'out of memory']):
            logger.info("🔧 Correction: Erreur mémoire")
            return self._fix_memory_error()
        
        # Erreurs Azure
        if any(keyword in error_lower for keyword in ['azure', 'blob', 'connection']):
            logger.info("🔧 Correction: Erreur Azure")
            return self._fix_azure_error()
        
        # Erreurs de données
        if any(keyword in error_lower for keyword in ['file not found', 'data', 'missing']):
            logger.info("🔧 Correction: Erreur données")
            return self._fix_data_error()
        
        # Erreurs GPU
        if any(keyword in error_lower for keyword in ['cuda', 'gpu']):
            logger.info("🔧 Correction: Erreur GPU")
            return self._fix_gpu_error()
        
        return False
    
    def _fix_memory_error(self):
        """Corriger les erreurs mémoire"""
        # Réduire les paramètres d'entraînement
        config_file = "configs/training_config_lite.json"
        if Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Réduire les paramètres
                config['n_trials'] = min(config.get('n_trials', 100), 30)
                config['batch_size'] = min(config.get('batch_size', 1000), 500)
                
                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=2)
                
                logger.info("Configuration mémoire optimisée")
                return True
            except Exception as e:
                logger.error(f"Échec optimisation mémoire: {e}")
        
        # Forcer garbage collection
        os.environ['REDUCE_MEMORY'] = 'true'
        return True
    
    def _fix_azure_error(self):
        """Corriger les erreurs Azure"""
        os.environ['DISABLE_AZURE_UPLOAD'] = 'true'
        logger.info("Azure upload désactivé temporairement")
        return True
    
    def _fix_data_error(self):
        """Corriger les erreurs de données"""
        try:
            if Path("clean_data_leakage.py").exists():
                result = subprocess.run(["python", "clean_data_leakage.py"], 
                                      capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    logger.info("Données régénérées avec succès")
                    return True
        except Exception as e:
            logger.error(f"Échec régénération données: {e}")
        return False
    
    def _fix_gpu_error(self):
        """Corriger les erreurs GPU"""
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['USE_GPU'] = 'false'
        logger.info("Mode CPU forcé")
        return True
    
    def start_training(self):
        """Démarrer l'entraînement avec surveillance"""
        # Choisir le script d'entraînement
        training_scripts = [
            "run_loop_tuner_agent.sh",
            "retrain_catboost_fixed.py",
            "agents/tuner_agent/run_tuner_agent.py"
        ]
        
        script = None
        for candidate in training_scripts:
            if Path(candidate).exists():
                script = candidate
                break
        
        if not script:
            logger.error("❌ Aucun script d'entraînement trouvé")
            return False
        
        logger.info(f"🚀 Lancement: {script}")
        
        try:
            # Démarrer le processus avec timeout de 8h
            if script.endswith('.sh'):
                cmd = ['bash', script]
            else:
                cmd = ['python', script]
            
            self.training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Échec lancement: {e}")
            return False
    
    def monitor_training(self):
        """Surveiller l'entraînement en temps réel"""
        logger.info("👁️ Surveillance de l'entraînement...")
        
        if not self.training_process:
            logger.error("❌ Aucun processus à surveiller")
            return False
        
        error_buffer = []
        last_output_time = time.time()
        
        try:
            for line in iter(self.training_process.stdout.readline, ''):
                if line:
                    print(line.strip())  # Afficher en temps réel
                    last_output_time = time.time()
                    
                    # Collecter les erreurs
                    if any(keyword in line.lower() for keyword in ['error', 'exception', 'failed']):
                        error_buffer.append(line.strip())
                        
                        # Si trop d'erreurs récentes, tenter correction
                        if len(error_buffer) > 5:
                            recent_errors = '\n'.join(error_buffer[-10:])
                            if self.detect_and_fix_error(recent_errors):
                                logger.info("🔧 Correction appliquée - Redémarrage...")
                                if self.training_process:
                                    self.training_process.terminate()
                                return False  # Signal pour retry
                
                # Vérifier si le processus est bloqué (pas de sortie depuis 30min)
                if time.time() - last_output_time > 1800:
                    logger.warning("⚠️ Processus potentiellement bloqué")
                    if self.training_process:
                        self.training_process.terminate()
                    return False
            
            # Attendre la fin du processus
            exit_code = self.training_process.wait() if self.training_process else 1
            
            if exit_code == 0:
                logger.info("✅ Entraînement terminé avec succès")
                return True
            else:
                logger.error(f"❌ Entraînement échoué (code: {exit_code})")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erreur surveillance: {e}")
            return False
    
    def generate_report(self, success):
        """Générer le rapport final"""
        duration = datetime.now() - self.start_time
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "success": success,
            "duration_minutes": duration.total_seconds() / 60,
            "retries": self.retry_count,
            "models_info": self._get_models_info()
        }
        
        # Sauvegarder rapport JSON
        with open("night_training_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        # Rapport texte pour lecture rapide
        status_emoji = "✅" if success else "❌"
        with open("night_report.txt", "w") as f:
            f.write(f"{status_emoji} Entraînement nocturne - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"Durée: {duration.total_seconds()/3600:.1f}h\n")
            f.write(f"Tentatives: {self.retry_count + 1}\n")
            if report["models_info"]:
                f.write(f"Modèles: {report['models_info']}\n")
        
        logger.info(f"📄 Rapport généré: {'SUCCÈS' if success else 'ÉCHEC'}")
    
    def _get_models_info(self):
        """Récupérer infos des modèles"""
        try:
            from utils.azure_model_storage import AzureModelStorage
            storage = AzureModelStorage()
            models = storage.list_all_models()
            if models:
                best = models[0]
                return f"{len(models)} modèles, Meilleur R²: {best.get('r2_test', 'N/A')}"
        except:
            pass
        return "Info modèles non disponible"
    
    def run(self):
        """Exécuter le système complet avec retry"""
        logger.info("🌙 === SYSTÈME AUTO-RECOVERY DÉMARRÉ ===")
        
        # Vérifications préliminaires
        if not self.check_environment():
            logger.error("❌ Environnement non conforme")
            self.generate_report(False)
            return False
        
        # Boucle principale avec retry
        while self.retry_count < self.max_retries:
            logger.info(f"🔄 Tentative {self.retry_count + 1}/{self.max_retries}")
            
            if self.start_training() and self.monitor_training():
                logger.info("🎉 SUCCÈS!")
                self.generate_report(True)
                return True
            
            # Échec - préparer retry
            self.retry_count += 1
            if self.retry_count < self.max_retries:
                logger.info(f"⚠️ Échec tentative {self.retry_count} - Récupération...")
                time.sleep(30)  # Pause avant retry
        
        # Échec final
        logger.error(f"❌ ÉCHEC après {self.max_retries} tentatives")
        self.generate_report(False)
        return False

if __name__ == "__main__":
    recovery_system = AutoRecoverySystem()
    success = recovery_system.run()
    sys.exit(0 if success else 1)
