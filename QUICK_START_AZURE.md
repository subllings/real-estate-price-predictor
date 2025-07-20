# 🚀 Guide de Déploiement Rapide - Tuner Agents sur Azure

## TL;DR - Déploiement en 3 commandes

```bash
# 1. Se connecter à Azure
az login

# 2. Lancer le déploiement automatisé
./deploy-azure-complete.sh

# 3. Accéder à votre application
# L'URL sera affichée à la fin du déploiement
```

## Qu'est-ce qui est déployé ?

✅ **React Frontend** containerisé (interface web)  
✅ **FastAPI Backend** avec endpoints tuner agents  
✅ **Azure Container Registry** pour les images Docker  
✅ **Azure Cosmos DB** pour le suivi des jobs  
✅ **Azure Container Instances** pour l'hébergement  

## Comment lancer un tuner agent depuis React ?

1. **Accéder à l'application** : Cliquer sur l'URL affichée après déploiement
2. **Navigation** : Menu → Model Training → New Training
3. **Configuration** :
   - **Model Type** : CatBoost+Optuna, XGBoost+Optuna, LightGBM+Optuna, Random Forest+Optuna, Stacked Ensemble+Optuna
   - **Termination Type** :
     - **Endless** : Pas de limite (arrêt manuel)
     - **Duration** : Durée en heures (ex: 2.5h)
     - **End Time** : Heure de fin (ex: 07:00)
     - **Max Trials** : Nombre max d'essais (ex: 100)
4. **Lancement** : Cliquer "Start Training"
5. **Suivi** : Le job apparaît dans la liste avec statut en temps réel

## Architecture Déployée

```
┌─────────────────┐    HTTP    ┌─────────────────┐
│  React Frontend │ ────────► │  FastAPI Backend│
│  (Port 80)      │           │  (Port 8002)    │
└─────────────────┘           └─────────────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │   Tuner Agents  │
                              │ (Background)    │
                              └─────────────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │   Cosmos DB     │
                              │ (Job Tracking)  │
                              └─────────────────┘
```

## Nouveaux Endpoints API

### 🚀 POST `/api/tuner/start`
Lance un tuner agent avec configuration personnalisée.

**Exemple de payload** :
```json
{
  "model_type": "catboost",
  "termination_type": "duration",
  "duration_hours": 2.5
}
```

### 📊 GET `/api/tuner/status/{job_id}`
Récupère le statut d'un job spécifique.

### 📋 GET `/api/tuner/list`
Liste tous les jobs tuner avec leur statut.

## Exemples de Configuration

### 🔄 Mode Endless (Formation continue)
```json
{
  "model_type": "xgboost",
  "termination_type": "endless"
}
```

### ⏱️ Mode Duration (Durée limitée)
```json
{
  "model_type": "lightgbm",
  "termination_type": "duration",
  "duration_hours": 3.0
}
```

### 🕐 Mode End Time (Heure de fin)
```json
{
  "model_type": "random_forest",
  "termination_type": "end_time",
  "end_time": "08:00",
  "stop_hour": 8,
  "stop_minute": 0
}
```

### 🎯 Mode Max Trials (Nombre d'essais)
```json
{
  "model_type": "stack_ensemble",
  "termination_type": "max_trials",
  "max_trials": 150
}
```

## Monitoring et Debug

### Voir les logs en temps réel
```bash
# Logs du backend (tuner agents)
az container logs --resource-group rg-real-estate-ml --name backend-api --follow

# Logs du frontend
az container logs --resource-group rg-real-estate-ml --name frontend-react --follow
```

### Vérifier les ressources
```bash
# Statut des containers
az container list --resource-group rg-real-estate-ml --output table

# Métriques Cosmos DB
az cosmosdb show --name cosmos-real-estate-ml --resource-group rg-real-estate-ml
```

### Redémarrer un container
```bash
# Redémarrer le backend
az container restart --resource-group rg-real-estate-ml --name backend-api

# Redémarrer le frontend
az container restart --resource-group rg-real-estate-ml --name frontend-react
```

## Nettoyage

```bash
# Supprimer toutes les ressources
az group delete --name rg-real-estate-ml --yes --no-wait
```

## Troubleshooting

### ❌ Problème : Container ne démarre pas
```bash
# Vérifier les logs d'erreur
az container logs --resource-group rg-real-estate-ml --name [CONTAINER_NAME]

# Vérifier les events
az container show --resource-group rg-real-estate-ml --name [CONTAINER_NAME]
```

### ❌ Problème : API non accessible
```bash
# Vérifier l'IP publique
az container show --resource-group rg-real-estate-ml --name backend-api --query ipAddress.ip --output tsv

# Tester l'endpoint
curl http://[BACKEND_IP]:8002/health
```

### ❌ Problème : Tuner agent ne se lance pas
- Vérifier que les scripts sont présents dans le container backend
- Vérifier les logs du backend pour les erreurs Python
- S'assurer que les paramètres sont valides

## Support Multi-Model

Le système supporte maintenant tous ces types de modèles :

| Model Type | Script | Description |
|------------|--------|-------------|
| `catboost` | CatBoostTuner | Gradient boosting avec gestion catégories |
| `xgboost` | XGBoostTuner | Extreme Gradient Boosting |
| `lightgbm` | LightGBMTuner | Light Gradient Boosting Machine |
| `random_forest` | RandomForestTuner | Forêt aléatoire |
| `stack_ensemble` | StackEnsembleTuner | Ensemble de modèles stackés |

## Coûts Azure Estimés

- **Container Instances** : ~€50-100/mois (selon utilisation)
- **Cosmos DB** : ~€25-50/mois (selon données)
- **Container Registry** : ~€5/mois
- **Stockage** : ~€5-10/mois

**Total estimé : €85-165/mois**

## Questions Fréquentes

**Q: Puis-je lancer plusieurs tuner agents en même temps ?**  
R: Oui ! Chaque job a un ID unique et s'exécute indépendamment.

**Q: Comment arrêter un tuner agent en cours ?**  
R: Via l'interface React ou en supprimant le job dans Cosmos DB.

**Q: Les modèles sont-ils sauvegardés ?**  
R: Oui, les meilleurs modèles sont sauvegardés automatiquement.

**Q: Puis-je personnaliser les hyperparamètres ?**  
R: Oui, modifiez les fichiers tuner dans le dossier `agents/`.

---

🎉 **Votre système de tuner agents est maintenant prêt sur Azure !**
