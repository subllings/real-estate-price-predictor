# 🚀 Guide Complet - Training Local depuis React

## 🎯 Vue d'ensemble

Votre interface React peut maintenant démarrer un **model training directement sur votre machine locale** via l'API ! Voici comment tout fonctionne :

## 🏗️ Architecture de la Solution

```
React Frontend (Interface utilisateur)
    ↓ HTTP POST /api/training/start-local
FastAPI Backend (Orchestrateur)
    ↓ Exécution Python locale
Entraînement ML Local (CatBoost, XGBoost, etc.)
    ↓ Sauvegarde et logs
Cosmos DB (Suivi en temps réel)
```

## 🔧 Ce qui a été ajouté

### 1. ✅ Nouveaux endpoints API Backend

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/api/training/start-local` | POST | Démarre un training local |
| `/api/training/local-status/{job_id}` | GET | Statut d'un job local |
| `/api/training/list-local` | GET | Liste tous les jobs locaux |

### 2. ✅ Interface React enrichie

- **Bouton "Local Training"** à côté de "New Training"
- **Modal de configuration** avec tous les paramètres
- **Intégration complète** avec l'API existante
- **Feedback temps réel** dans l'interface

### 3. ✅ Entraînement ML complet

- **5 types de modèles** : CatBoost, XGBoost, LightGBM, Random Forest, Neural Network
- **Preprocessing automatique** : Gestion des variables catégorielles
- **Cross-validation** : Validation croisée configurable
- **Métriques complètes** : R², RMSE, MAE, scores CV
- **Sauvegarde automatique** : Modèles sauvés au format .pkl

## 🎮 Comment utiliser

### 1. Démarrer l'API Backend
```bash
cd app/backend-api-price-prediction
python main.py
```
L'API sera disponible sur `http://localhost:8002`

### 2. Démarrer le Frontend React
```bash
cd app/frontend-react
npm start
```
L'interface sera disponible sur `http://localhost:3000`

### 3. Lancer un Training Local

1. **Aller dans Model Training** → onglet "Training Pipeline"
2. **Cliquer "Local Training"** (bouton violet)
3. **Configurer le training** :
   - **Model Type** : CatBoost, XGBoost, LightGBM, Random Forest, Neural Network
   - **Dataset Path** : Chemin vers votre fichier CSV (ex: `data/processed/train.csv`)
   - **Target Column** : Nom de la colonne cible (ex: `price`)
   - **Training Config** : Test size, CV folds, random state
   - **Hyperparameter Tuning** : Activer/désactiver, nombre d'essais, timeout
   - **Output Config** : Sauvegarder le modèle, nom personnalisé

4. **Lancer** : Cliquer "Start Local Training"
5. **Suivre** : Le job apparaît dans la liste avec progression temps réel

## 📊 Configuration Exemple

### Configuration Rapide (CatBoost - 30 min)
```json
{
  "model_type": "catboost",
  "dataset_path": "data/processed/train.csv",
  "target_column": "price",
  "training_config": {
    "test_size": 0.2,
    "cv_folds": 5
  },
  "hyperparameter_tuning": {
    "enabled": true,
    "n_trials": 50,
    "timeout_minutes": 30
  },
  "output_config": {
    "save_model": true,
    "model_name": "catboost_quick_v1"
  }
}
```

### Configuration Approfondie (XGBoost - 2h)
```json
{
  "model_type": "xgboost",
  "dataset_path": "data/processed/train.csv",
  "target_column": "price",
  "training_config": {
    "test_size": 0.2,
    "cv_folds": 10
  },
  "hyperparameter_tuning": {
    "enabled": true,
    "n_trials": 200,
    "timeout_minutes": 120
  },
  "output_config": {
    "save_model": true,
    "model_name": "xgboost_deep_v1"
  }
}
```

## 🔍 Suivi et Monitoring

### Dans l'interface React
- **Progression en pourcentage** : 0% → 100%
- **Étape actuelle** : "loading_data", "preprocessing_data", "training_model", "evaluating_model", "completed"
- **Métriques temps réel** : R² train/test, RMSE, MAE
- **Temps estimé** : Basé sur la configuration

### Dans les logs
```bash
# Logs détaillés du backend
tail -f logs/backend.log

# Ou directement dans la console
python main.py
```

## 📈 Métriques Retournées

Après training, vous obtenez :

```json
{
  "metrics": {
    "train_r2": 0.8945,
    "test_r2": 0.8512,
    "train_rmse": 45623.21,
    "test_rmse": 52341.87,
    "train_mae": 32145.66,
    "test_mae": 38901.23,
    "cv_mean_r2": 0.8423,
    "cv_std_r2": 0.0156,
    "overfitting_gap": 0.0433
  },
  "model_path": "models/catboost_quick_v1.pkl",
  "generalization_score": 0.9514
}
```

## 🛠️ Types de Modèles Supportés

### 1. CatBoost
- **Avantages** : Gestion native des catégories, robuste
- **Temps** : Rapide à moyen
- **Utilisation** : Recommandé pour débuter

### 2. XGBoost  
- **Avantages** : Très performant, beaucoup d'options
- **Temps** : Moyen à long
- **Utilisation** : Compétitions, production

### 3. LightGBM
- **Avantages** : Le plus rapide, économe en mémoire
- **Temps** : Très rapide
- **Utilisation** : Gros datasets, prototypage

### 4. Random Forest
- **Avantages** : Simple, interpretable, robuste
- **Temps** : Rapide
- **Utilisation** : Baseline, features importantes

### 5. Neural Network (Basique)
- **Avantages** : Flexible, non-linéaire
- **Temps** : Variable
- **Utilisation** : Patterns complexes

## 🔧 Personnalisation Avancée

### Modifier les hyperparamètres par défaut
Éditez le code dans `run_local_training_process()` :

```python
# Pour CatBoost
model = CatBoostRegressor(
    iterations=2000,        # ← Augmenter pour plus de précision
    learning_rate=0.05,     # ← Diminuer pour plus de stabilité
    depth=8,                # ← Augmenter pour plus de complexité
    random_seed=42,
    verbose=False
)
```

### Ajouter un nouveau type de modèle
1. **Backend** : Ajouter le modèle dans `run_local_training_process()`
2. **Frontend** : Ajouter dans `modelTypes` du modal
3. **API** : Ajouter dans `valid_models`

### Modifier le preprocessing
Personnalisez la section preprocessing dans `run_local_training_process()` :

```python
# Exemple : Normalisation des features numériques
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

## 🚨 Gestion d'Erreurs

### Erreurs communes et solutions

| Erreur | Cause | Solution |
|--------|-------|----------|
| "Dataset non trouvé" | Chemin incorrect | Vérifier le path relatif depuis backend-api-price-prediction/ |
| "Colonne target non trouvée" | Nom de colonne incorrect | Vérifier les noms de colonnes dans votre CSV |
| "Module non trouvé" | Dépendance manquante | `pip install catboost xgboost lightgbm scikit-learn` |
| "Timeout" | Training trop long | Réduire n_trials ou timeout_minutes |

### Debug avancé
```bash
# Tester l'endpoint directement
curl -X POST http://localhost:8002/api/training/start-local \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "catboost",
    "dataset_path": "data/processed/train.csv",
    "target_column": "price"
  }'

# Voir le statut d'un job
curl http://localhost:8002/api/training/local-status/local_training_catboost_1234567890
```

## 🎯 Workflow Recommandé

### 1. **Prototypage Rapide**
```
LightGBM → 30 trials → 15 minutes
```

### 2. **Modèle de Production**  
```
CatBoost → 100 trials → 60 minutes
```

### 3. **Optimisation Poussée**
```
XGBoost → 200 trials → 120 minutes
```

### 4. **Ensemble Final**
```
Combiner les 3 meilleurs modèles
```

## 🔄 Intégration avec le Workflow Existant

### Complémentaire aux Tuner Agents
- **Tuner Agents** : Optimisation continue en arrière-plan
- **Local Training** : Training ponctuel avec configuration spécifique
- **Les deux** utilisent le même système de suivi Cosmos DB

### Réutilisation des modèles
```python
# Charger un modèle sauvé
import joblib
model = joblib.load('models/catboost_quick_v1.pkl')

# Faire des prédictions
predictions = model.predict(new_data)
```

## 🎉 Avantages de cette Approche

✅ **Contrôle total** : Tous les paramètres configurables via l'interface
✅ **Suivi complet** : Progression, métriques, logs en temps réel  
✅ **Intégration native** : Utilise votre infrastructure existante
✅ **Flexibilité** : Support de 5 types de modèles différents
✅ **Sauvegarde automatique** : Modèles et métriques persistés
✅ **Interface intuitive** : Configuration via formulaire React
✅ **Monitoring avancé** : Cosmos DB pour le suivi multi-machines

## 🚀 Résultat Final

Vous pouvez maintenant :
1. **Cliquer "Local Training"** dans votre interface React
2. **Configurer** votre modèle en quelques clics
3. **Lancer** l'entraînement sur votre machine
4. **Suivre** la progression en temps réel
5. **Récupérer** un modèle prêt à l'emploi

**C'est exactement ce que vous vouliez ! 🎯**
