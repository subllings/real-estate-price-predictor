#!/usr/bin/env python3
"""
Système de synchronisation distributed pour entraînement ML multi-machines
Permet de continuer l'entraînement sur laptop si desktop s'arrête
"""
import os
import json
import time
import socket
import threading
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

class DistributedTrainingManager:
    """
    Gestionnaire d'entraînement distribué avec handover automatique
    """
    
    def __init__(self, machine_role="auto"):
        self.machine_id = self._get_machine_id()
        self.machine_role = self._determine_role(machine_role)
        self.status_file = "distributed_training_status.json"
        self.heartbeat_interval = 30  # 30 secondes
        self.master_timeout = 120  # 2 minutes sans heartbeat = considéré mort
        self.running = False
        
        print(f"🖥️ Machine ID: {self.machine_id}")
        print(f"🎭 Rôle: {self.machine_role}")
    
    def _get_machine_id(self):
        """Générer un ID unique pour cette machine"""
        hostname = socket.gethostname()
        # Utiliser le nom de la machine + quelques caractères de hash pour unicité
        machine_hash = hashlib.md5(hostname.encode()).hexdigest()[:8]
        return f"{hostname}_{machine_hash}"
    
    def _determine_role(self, requested_role):
        """Déterminer le rôle de cette machine"""
        if requested_role in ["master", "slave"]:
            return requested_role
        
        # Auto-détection basée sur la puissance/type de machine
        hostname = socket.gethostname().lower()
        
        # Desktop généralement plus puissant = master par défaut
        if any(keyword in hostname for keyword in ["desktop", "pc", "gaming", "workstation"]):
            return "master"
        elif any(keyword in hostname for keyword in ["laptop", "notebook", "mobile"]):
            return "slave"
        
        # Par défaut, première machine à démarrer = master
        if not Path(self.status_file).exists():
            return "master"
        else:
            return "slave"
    
    def update_status(self, training_status="idle", progress=0, current_trial=0, best_r2=0):
        """Mettre à jour le statut de l'entraînement"""
        status = {
            "machine_id": self.machine_id,
            "machine_role": self.machine_role,
            "last_heartbeat": datetime.now().isoformat(),
            "training_status": training_status,
            "progress_percent": progress,
            "current_trial": current_trial,
            "best_r2": best_r2,
            "azure_synced": self._check_azure_sync(),
            "pid": os.getpid(),
            "start_time": getattr(self, 'start_time', datetime.now().isoformat())
        }
        
        # Si fichier existe, merger avec les autres machines
        existing_status = {}
        if Path(self.status_file).exists():
            try:
                with open(self.status_file, 'r') as f:
                    existing_status = json.load(f)
            except:
                existing_status = {}
        
        # Mettre à jour notre machine
        if "machines" not in existing_status:
            existing_status["machines"] = {}
        
        existing_status["machines"][self.machine_id] = status
        existing_status["last_update"] = datetime.now().isoformat()
        
        # Déterminer qui est le master actuel
        existing_status["current_master"] = self._determine_current_master(existing_status["machines"])
        
        with open(self.status_file, 'w') as f:
            json.dump(existing_status, f, indent=2)
    
    def _determine_current_master(self, machines):
        """Déterminer qui est le master actuel"""
        now = datetime.now()
        active_machines = []
        
        for machine_id, machine_info in machines.items():
            last_heartbeat = datetime.fromisoformat(machine_info["last_heartbeat"])
            if (now - last_heartbeat).total_seconds() < self.master_timeout:
                active_machines.append((machine_id, machine_info))
        
        # Priorité: master déclaré > machine la plus récente
        for machine_id, info in active_machines:
            if info["machine_role"] == "master" and info["training_status"] != "idle":
                return machine_id
        
        # Sinon, machine avec le plus de progrès
        if active_machines:
            active_machines.sort(key=lambda x: x[1]["progress_percent"], reverse=True)
            return active_machines[0][0]
        
        return self.machine_id
    
    def _check_azure_sync(self):
        """Vérifier si Azure est synchronisé"""
        try:
            from utils.azure_model_storage import AzureModelStorage
            storage = AzureModelStorage()
            models = storage.list_all_models()
            return len(models) > 0
        except:
            return False
    
    def should_i_be_master(self):
        """Déterminer si cette machine doit devenir master"""
        if not Path(self.status_file).exists():
            return self.machine_role == "master"
        
        try:
            with open(self.status_file, 'r') as f:
                status = json.load(f)
            
            current_master = status.get("current_master")
            
            # Si pas de master ou master inactif
            if not current_master or current_master == self.machine_id:
                return True
            
            # Vérifier si le master actuel est toujours actif
            if current_master in status.get("machines", {}):
                master_info = status["machines"][current_master]
                last_heartbeat = datetime.fromisoformat(master_info["last_heartbeat"])
                if (datetime.now() - last_heartbeat).total_seconds() > self.master_timeout:
                    print(f"⚠️ Master {current_master} inactif depuis {self.master_timeout}s")
                    return True
            
            return False
            
        except Exception as e:
            print(f"❌ Erreur vérification master: {e}")
            return self.machine_role == "master"
    
    def get_training_state(self):
        """Récupérer l'état actuel de l'entraînement depuis Azure/fichiers"""
        training_state = {
            "last_trial": 0,
            "best_params": {},
            "best_r2": 0,
            "completed_trials": [],
            "azure_models": []
        }
        
        try:
            # État depuis Azure
            from utils.azure_model_storage import AzureModelStorage
            storage = AzureModelStorage()
            models = storage.list_all_models()
            
            if models:
                training_state["azure_models"] = models
                training_state["best_r2"] = max(m.get("r2_test", 0) for m in models)
                training_state["last_trial"] = len(models)
        except Exception as e:
            print(f"⚠️ Impossible de récupérer l'état Azure: {e}")
        
        try:
            # État depuis les logs locaux si disponible
            if Path("training_logs").exists():
                training_state["completed_trials"] = len(list(Path("training_logs").glob("*.json")))
        except Exception as e:
            print(f"⚠️ Impossible de récupérer l'état local: {e}")
        
        return training_state
    
    def start_heartbeat(self):
        """Démarrer le système de heartbeat"""
        def heartbeat_loop():
            while self.running:
                try:
                    # Déterminer statut d'entraînement
                    if hasattr(self, 'training_progress') and self.training_progress:
                        status = "training"
                        progress = self.training_progress.get("progress", 0)
                        trial = self.training_progress.get("trial", 0)
                        best_r2 = self.training_progress.get("best_r2", 0)
                    else:
                        status = "monitoring"
                        progress = 0
                        trial = 0
                        best_r2 = 0
                    
                    self.update_status(status, progress, trial, best_r2)
                    time.sleep(self.heartbeat_interval)
                except Exception as e:
                    print(f"⚠️ Erreur heartbeat: {e}")
                    time.sleep(self.heartbeat_interval)
        
        self.running = True
        self.heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()
        print(f"💓 Heartbeat démarré (interval: {self.heartbeat_interval}s)")
    
    def stop_heartbeat(self):
        """Arrêter le heartbeat"""
        self.running = False
        if hasattr(self, 'heartbeat_thread'):
            self.heartbeat_thread.join(timeout=5)
    
    def wait_for_master_clearance(self, timeout=300):
        """Attendre que le master actuel termine ou disparaisse"""
        print("⏳ Attente de la libération du master...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.should_i_be_master():
                print("✅ Autorisation de devenir master")
                return True
            
            # Afficher le statut du master actuel
            try:
                with open(self.status_file, 'r') as f:
                    status = json.load(f)
                
                current_master = status.get("current_master")
                if current_master and current_master in status.get("machines", {}):
                    master_info = status["machines"][current_master]
                    print(f"🖥️ Master actuel: {current_master} - {master_info['training_status']}")
                
            except:
                pass
            
            time.sleep(30)
        
        print("⏰ Timeout attente master - Prise de contrôle forcée")
        return True
    
    def handover_to_slave(self):
        """Transférer le contrôle à une machine slave"""
        print("🔄 Transfert de contrôle à une machine slave...")
        
        try:
            with open(self.status_file, 'r') as f:
                status = json.load(f)
            
            # Trouver une machine slave active
            slaves = []
            for machine_id, machine_info in status.get("machines", {}).items():
                if (machine_id != self.machine_id and 
                    machine_info.get("machine_role") == "slave"):
                    
                    last_heartbeat = datetime.fromisoformat(machine_info["last_heartbeat"])
                    if (datetime.now() - last_heartbeat).total_seconds() < self.master_timeout:
                        slaves.append((machine_id, machine_info))
            
            if slaves:
                # Choisir le slave le plus récent
                slaves.sort(key=lambda x: x[1]["last_heartbeat"], reverse=True)
                new_master = slaves[0][0]
                
                print(f"🤝 Transfert vers: {new_master}")
                
                # Marquer le transfert
                self.update_status("handover_complete", 100, 0, 0)
                
                # Créer un signal pour le nouveau master
                handover_signal = {
                    "from_master": self.machine_id,
                    "to_master": new_master,
                    "timestamp": datetime.now().isoformat(),
                    "training_state": self.get_training_state()
                }
                
                with open("handover_signal.json", "w") as f:
                    json.dump(handover_signal, f, indent=2)
                
                return True
            else:
                print("⚠️ Aucune machine slave disponible pour le transfert")
                return False
                
        except Exception as e:
            print(f"❌ Erreur durante handover: {e}")
            return False
    
    def check_for_handover_signal(self):
        """Vérifier s'il y a un signal de transfert pour cette machine"""
        handover_file = "handover_signal.json"
        
        if Path(handover_file).exists():
            try:
                with open(handover_file, 'r') as f:
                    signal = json.load(f)
                
                if signal.get("to_master") == self.machine_id:
                    print(f"📨 Signal de transfert reçu de {signal.get('from_master')}")
                    
                    # Récupérer l'état de l'entraînement
                    training_state = signal.get("training_state", {})
                    
                    # Supprimer le signal après lecture
                    os.remove(handover_file)
                    
                    return training_state
                    
            except Exception as e:
                print(f"⚠️ Erreur lecture signal handover: {e}")
        
        return None


def create_distributed_training_script():
    """Créer le script de lancement distribué"""
    script_content = '''#!/usr/bin/env python3
"""
Script de lancement d'entraînement distribué
Usage: python distributed_training_launcher.py [master|slave|auto]
"""
import sys
import time
from distributed_training_manager import DistributedTrainingManager

def main():
    role = sys.argv[1] if len(sys.argv) > 1 else "auto"
    
    print("🌐 SYSTÈME D'ENTRAÎNEMENT DISTRIBUÉ")
    print("=" * 40)
    
    manager = DistributedTrainingManager(machine_role=role)
    manager.start_heartbeat()
    
    try:
        # Vérifier s'il y a un signal de transfert
        handover_state = manager.check_for_handover_signal()
        if handover_state:
            print("🎯 Reprise d'entraînement après transfert")
        
        # Déterminer si cette machine doit être master
        if manager.should_i_be_master():
            print("👑 Mode MASTER - Démarrage entraînement")
            
            # Lancer l'entraînement principal
            from auto_recovery_system import AutoRecoverySystem
            recovery_system = AutoRecoverySystem()
            
            # Intégrer le monitoring distribué
            original_start_training = recovery_system.start_training
            def distributed_start_training():
                manager.training_progress = {"progress": 0, "trial": 0, "best_r2": 0}
                return original_start_training()
            
            recovery_system.start_training = distributed_start_training
            
            success = recovery_system.run()
            
            if not success:
                print("⚠️ Entraînement échoué - Tentative de transfert")
                manager.handover_to_slave()
        
        else:
            print("🔄 Mode SLAVE - Surveillance et attente")
            
            # Mode surveillance, attendre de devenir master
            while True:
                if manager.should_i_be_master():
                    print("🚀 Promotion en MASTER - Démarrage entraînement")
                    
                    # Récupérer l'état et continuer l'entraînement
                    training_state = manager.get_training_state()
                    print(f"📊 État récupéré: {len(training_state.get('azure_models', []))} modèles")
                    
                    # Lancer l'entraînement avec état restauré
                    from auto_recovery_system import AutoRecoverySystem
                    recovery_system = AutoRecoverySystem()
                    success = recovery_system.run()
                    
                    if not success:
                        manager.handover_to_slave()
                
                time.sleep(30)  # Vérifier toutes les 30 secondes
    
    except KeyboardInterrupt:
        print("\\n🛑 Arrêt demandé")
        if manager.machine_role == "master":
            manager.handover_to_slave()
    
    finally:
        manager.stop_heartbeat()
        manager.update_status("stopped", 100, 0, 0)

if __name__ == "__main__":
    main()
'''
    
    with open("distributed_training_launcher.py", "w") as f:
        f.write(script_content)
    
    print("✅ Script de lancement distribué créé: distributed_training_launcher.py")


if __name__ == "__main__":
    # Créer le script de lancement
    create_distributed_training_script()
    
    # Test du système
    manager = DistributedTrainingManager()
    print(f"Test: Machine {manager.machine_id} - Rôle: {manager.machine_role}")
