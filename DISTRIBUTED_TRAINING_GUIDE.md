# 🌐 GUIDE D'ENTRAÎNEMENT DISTRIBUÉ

## 🎯 Objectif
Permettre l'entraînement ML en continu entre **desktop** et **laptop** avec handover automatique.

## 🚀 Démarrage Rapide

### 📋 Prérequis
- Azure Storage configuré sur les deux machines
- Code synchronisé (git ou copie manuelle)
- Python + requirements installés

### 🖥️ Sur le Desktop (Master)
```bash
# Lancer en master principal
./launch_desktop_master.sh

# Ou version Windows
launch_desktop_master.bat
```

### 💻 Sur le Laptop (Slave/Backup)
```bash
# Lancer en mode surveillance
./launch_laptop_slave.sh

# Ou version Windows  
launch_laptop_slave.bat
```

## 🔄 Scénarios d'Utilisation

### Scénario 1: Desktop + Laptop en parallèle
1. **Desktop**: `./launch_desktop_master.sh` (entraînement principal)
2. **Laptop**: `./launch_laptop_slave.sh` (surveillance)
3. Si desktop s'arrête → laptop prend le relais automatiquement

### Scénario 2: Laptop seul (desktop HS)
1. **Laptop**: `./launch_laptop_slave.sh`
2. Choisir option **2** (Force Master)
3. Laptop devient master immédiatement

### Scénario 3: Handover manuel
1. Sur desktop: **Ctrl+C** (arrêt propre)
2. Desktop transfère automatiquement vers laptop
3. Laptop reprend l'entraînement

## 📊 Monitoring

### Fichiers de Status
- `distributed_training_status.json` - État temps réel des machines
- `handover_signal.json` - Signals de transfert
- `night_report.txt` - Rapport final

### Logs
- `desktop_master_YYYYMMDD_HHMMSS.log` - Logs desktop
- `laptop_slave_YYYYMMDD_HHMMSS.log` - Logs laptop
- `auto_recovery_system.log` - Système de récupération

## 🔧 Configuration

### Variables d'Environnement
```bash
MACHINE_ROLE=master|slave
MACHINE_TYPE=desktop|laptop
TRAINING_PRIORITY=high|medium|low
```

### Timing
- **Heartbeat**: 30 secondes
- **Timeout Master**: 2 minutes
- **Training Timeout**: 8 heures

## 🆘 Dépannage

### Desktop ne démarre pas
```bash
# Vérifier prérequis
python -c "from utils.azure_model_storage import AzureModelStorage; print('OK')"

# Nettoyer statut
rm distributed_training_status.json
```

### Laptop ne voit pas le desktop
```bash
# Forcer en master
./launch_laptop_slave.sh
# Puis choisir option 2
```

### Synchronisation
```bash
# Synchroniser le code
./sync_machines.sh auto

# Ou manuellement via git
git pull origin main
```

## 📈 Architecture Technique

### Auto-Recovery System
- Détection d'erreurs intelligente
- Corrections automatiques
- 3 tentatives avec fallbacks

### Distributed Manager
- Heartbeat entre machines
- Élection automatique du master
- Transfert de contrôle fluide

### Azure Integration
- Stockage automatique des modèles
- Synchronisation des métadonnées
- État partagé en temps réel

## 🎉 Résultat Attendu

```
📊 RAPPORT FINAL
================
✅ Entraînement terminé avec succès
🏆 Meilleur modèle: R² = 0.8542
📦 Modèles uploadés: 15/15
⏱️ Durée totale: 6h 23min
🔄 Handovers: 1 (desktop → laptop)
```

## 💡 Conseils

1. **Testez d'abord** avec un entraînement court
2. **Synchronisez** avant de dormir
3. **Vérifiez Azure** sur les deux machines
4. **Gardez les laptops branchés** 🔌
5. **Le matin**: vérifiez `night_report.txt`

---
*Système créé pour un entraînement ML 24/7 sans interruption* 🚀
