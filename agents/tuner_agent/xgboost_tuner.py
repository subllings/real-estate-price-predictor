import sys, os
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import optuna
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import logging
import traceback
import warnings
import time
from typing import Dict, Any, Optional

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supprimer les warnings de XGBoost
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

class XGBoostTuner:
    def __init__(self, X, y, n_trials=50, n_splits=3, early_stopping_rounds=20, use_gpu=False, optuna_params=None, random_state=42, 
                 model_evaluator=None, cosmos_logger=None, model_saver=None, feature_selection_method="all_features"):
        self.X = X
        self.y = y
        self.n_trials = n_trials
        self.n_splits = n_splits  # Réduit à 3 pour la stabilité
        self.early_stopping_rounds = early_stopping_rounds
        self.use_gpu = use_gpu  # Désactivé par défaut pour éviter les segfaults
        self.random_state = random_state
        self.best_params = None
        self.feature_selection_method = feature_selection_method  # Ajout pour logging
        
        # Composants externes
        self.model_evaluator = model_evaluator
        self.cosmos_logger = cosmos_logger
        self.model_saver = model_saver
        
        # Si optuna_params est fourni, l'utiliser directement (vient de l'API LLM)
        # Sinon utiliser les paramètres par défaut
        if optuna_params is not None:
            # Si optuna_params contient une structure avec "param_space", l'extraire
            if isinstance(optuna_params, dict) and "param_space" in optuna_params:
                self.optuna_params = optuna_params["param_space"]
                print(f"[INFO] Using parameter space from API (extracted): {list(self.optuna_params.keys())}")
            else:
                self.optuna_params = optuna_params
                print(f"[INFO] Using parameter space from API: {list(optuna_params.keys())}")
        else:
            self.optuna_params = self._get_default_params()
            print(f"[INFO] Using default parameter space: {list(self.optuna_params.keys())}")
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Retourne les paramètres par défaut pour XGBoost"""
        return {
            "learning_rate": (0.01, 0.3),
            "max_depth": (3, 10),
            "min_child_weight": (1.0, 10.0),
            "subsample": (0.5, 1.0),
            "colsample_bytree": (0.5, 1.0),
            "gamma": (0.0, 5.0),
            "reg_alpha": (0.0, 1.0),
            "reg_lambda": (0.0, 1.0)
        }

    def validate_trial_params(self, params: Dict[str, Any]) -> bool:
        """Valide que les paramètres du trial sont compatibles avec XGBoost"""
        try:
            # Paramètres essentiels requis pour XGBoost
            essential_params = {"learning_rate", "max_depth"}
            
            # Vérifier que les paramètres essentiels sont présents
            param_keys = set(params.keys()) - {"n_estimators", "tree_method", "verbosity", "random_state"}
            
            missing_essential = essential_params - param_keys
            if missing_essential:
                logger.error(f"Missing essential parameters in trial: {missing_essential}")
                return False
            
            # Les autres paramètres peuvent être optionnels
            expected_params = set(self.optuna_params.keys())
            missing_optional = expected_params - param_keys
            if missing_optional:
                logger.info(f"Optional parameters not provided: {missing_optional} - using defaults")
            
            # Vérifier la validité des valeurs
            if params.get("learning_rate", 0) <= 0:
                logger.warning("learning_rate must be positive")
                return False
                
            if params.get("max_depth", 0) <= 0:
                logger.warning("max_depth must be positive")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating trial parameters: {e}")
            return False

    def objective(self, trial):
        """Fonction objectif pour Optuna avec gestion d'erreurs robuste"""
        trial_start_time = time.time()  # Début du trial
        
        try:
            # Construire les paramètres dynamiquement selon l'espace de paramètres fourni
            params = {}
            
            # Paramètres d'optimisation dynamiques
            for param_name, param_config in self.optuna_params.items():
                try:
                    # Gérer le format API LLM (avec "method", "low", "high", etc.)
                    if isinstance(param_config, dict) and "method" in param_config:
                        method = param_config["method"]
                        
                        if method == "suggest_float":
                            params[param_name] = trial.suggest_float(
                                param_name, 
                                param_config["low"], 
                                param_config["high"]
                            )
                        elif method == "suggest_int":
                            params[param_name] = trial.suggest_int(
                                param_name, 
                                param_config["low"], 
                                param_config["high"]
                            )
                        elif method == "suggest_categorical":
                            params[param_name] = trial.suggest_categorical(
                                param_name, 
                                param_config["choices"]
                            )
                        else:
                            logger.warning(f"Unknown method '{method}' for parameter '{param_name}', skipping...")
                    
                    # Gérer le format tuple simple (low, high) pour compatibilité
                    elif isinstance(param_config, (tuple, list)) and len(param_config) == 2:
                        if param_name == "learning_rate":
                            params[param_name] = trial.suggest_float(param_name, param_config[0], param_config[1])
                        elif param_name == "max_depth":
                            params[param_name] = trial.suggest_int(param_name, param_config[0], param_config[1])
                        elif param_name in ["min_child_weight", "subsample", "colsample_bytree", "colsample_bylevel", "colsample_bynode", "gamma", "reg_alpha", "reg_lambda"]:
                            params[param_name] = trial.suggest_float(param_name, param_config[0], param_config[1])
                        elif param_name == "n_estimators":
                            params[param_name] = trial.suggest_int(param_name, param_config[0], param_config[1])
                        elif param_name == "max_delta_step":
                            params[param_name] = trial.suggest_int(param_name, param_config[0], param_config[1])
                        else:
                            logger.warning(f"Unknown parameter: {param_name}, skipping...")
                    
                    else:
                        logger.warning(f"Invalid config for parameter '{param_name}': {param_config}, skipping...")
                        
                except Exception as param_error:
                    logger.error(f"Error processing parameter '{param_name}': {param_error}")
                    continue
            
            # Gestion spéciale pour max_leaves (dépend de grow_policy)
            if "grow_policy" in params and params["grow_policy"] == "lossguide":
                max_leaves_config = self.optuna_params.get("max_leaves")
                if max_leaves_config and isinstance(max_leaves_config, dict) and max_leaves_config.get("method") == "suggest_int":
                    params["max_leaves"] = trial.suggest_int(
                        "max_leaves", 
                        max_leaves_config["low"], 
                        max_leaves_config["high"]
                    )
            
            # Paramètres fixes
            params.update({
                "n_estimators": params.get("n_estimators", 1000),
                "tree_method": "auto",  # Forcer CPU pour éviter les segfaults
                "verbosity": 0,
                "random_state": self.random_state
            })
            
            # Valider les paramètres
            if not self.validate_trial_params(params):
                logger.warning(f"Invalid parameters for trial {trial.number}")
                return float('inf')
            
            logger.info(f"Trial {trial.number}: Testing parameters {params}")
            
            # Cross-validation avec gestion d'erreurs et métriques détaillées
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            scores = []
            training_times = []
            best_iterations = []

            for fold_idx, (train_idx, valid_idx) in enumerate(kf.split(self.X)):
                try:
                    fold_start_time = time.time()
                    
                    X_train, X_valid = self.X.iloc[train_idx], self.X.iloc[valid_idx]
                    y_train, y_valid = self.y.iloc[train_idx], self.y.iloc[valid_idx]

                    # Configure early stopping in model initialization for XGBoost 2.0+
                    model_params = params.copy()
                    model_params['early_stopping_rounds'] = self.early_stopping_rounds
                    model_params['eval_metric'] = 'rmse'
                    
                    model = XGBRegressor(**model_params)
                    
                    # Entraînement avec validation set (early stopping configuré dans le modèle)
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_valid, y_valid)],
                        verbose=False
                    )

                    fold_end_time = time.time()
                    fold_training_time = fold_end_time - fold_start_time
                    training_times.append(fold_training_time)

                    preds = model.predict(X_valid)
                    rmse = np.sqrt(mean_squared_error(y_valid, preds))
                    scores.append(rmse)
                    
                    # Récupérer la meilleure itération (nombre d'arbres utilisés)
                    best_iteration = getattr(model, 'best_iteration', params.get('n_estimators', 1000))
                    best_iterations.append(best_iteration)
                    
                    logger.debug(f"Fold {fold_idx + 1}/{self.n_splits}: RMSE = {rmse:.4f}, "
                               f"Training time = {fold_training_time:.2f}s, Best iteration = {best_iteration}")
                    
                except Exception as fold_error:
                    logger.error(f"Error in fold {fold_idx + 1}: {fold_error}")
                    # En cas d'erreur sur un fold, retourner une pénalité
                    return float('inf')

            if not scores:
                logger.error("No valid scores obtained from cross-validation")
                return float('inf')
            
            # Calculer les métriques agrégées
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            mean_training_time = np.mean(training_times)
            mean_best_iteration = np.mean(best_iterations)
            trial_duration = time.time() - trial_start_time
            
            logger.info(f"Trial {trial.number}: Mean RMSE = {mean_score:.4f} ± {std_score:.4f}, "
                       f"Avg training time = {mean_training_time:.2f}s, "
                       f"Avg best iteration = {mean_best_iteration:.0f}")
            
            # Logger dans Cosmos DB si disponible avec métriques enrichies
            if self.cosmos_logger:
                try:
                    trial_info = {
                        "trial_number": trial.number,
                        "params": params,
                        "mean_rmse": mean_score,
                        "std_rmse": std_score,
                        "individual_scores": scores,
                        "model_type": "xgboost",
                        
                        # 🚀 Nouvelles métriques ajoutées
                        "training_time_seconds": mean_training_time,
                        "trial_duration_seconds": trial_duration,
                        "n_features_used": len(self.X.columns),
                        "feature_selection_method": self.feature_selection_method,
                        "cv_strategy": f"KFold(n_splits={self.n_splits})",
                        "early_stopping_rounds": self.early_stopping_rounds,
                        "eval_metric_used": "rmse",
                        "best_iteration": mean_best_iteration,
                        "best_iterations_by_fold": best_iterations,
                        "training_times_by_fold": training_times
                    }
                    self.cosmos_logger.log_trial(trial_info)
                except Exception as log_error:
                    logger.warning(f"Failed to log trial to Cosmos DB: {log_error}")

            return mean_score
            
        except Exception as e:
            trial_duration = time.time() - trial_start_time
            logger.error(f"Critical error in trial {trial.number} after {trial_duration:.2f}s: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return float('inf')

    def run_study(self):
        """Lance l'étude d'optimisation avec gestion d'erreurs"""
        study_start_time = time.time()
        
        try:
            logger.info(f"Starting XGBoost hyperparameter optimization with {self.n_trials} trials")
            logger.info(f"Parameter space: {self.optuna_params}")
            logger.info(f"Using GPU: {self.use_gpu}")
            logger.info(f"Feature selection method: {self.feature_selection_method}")
            logger.info(f"Number of features: {len(self.X.columns)}")
            logger.info(f"CV strategy: KFold(n_splits={self.n_splits})")
            
            # Créer l'étude Optuna
            study = optuna.create_study(direction="minimize")
            
            # Optimiser avec gestion d'erreurs
            study.optimize(self.objective, n_trials=self.n_trials)
            
            study_duration = time.time() - study_start_time
            
            # Récupérer les meilleurs paramètres
            self.best_params = study.best_params.copy()
            
            # Ajouter les paramètres fixes
            self.best_params.update({
                "n_estimators": self.best_params.get("n_estimators", 1000),
                "tree_method": "auto",  # Forcer CPU
                "verbosity": 0,
                "random_state": self.random_state
            })
            
            logger.info(f"Best parameters found: {self.best_params}")
            logger.info(f"Best score (RMSE): {study.best_value:.4f}")
            logger.info(f"Total study duration: {study_duration:.2f} seconds")
            
            # Évaluer le meilleur modèle si l'évaluateur est disponible
            if self.model_evaluator and self.best_params:
                try:
                    eval_start_time = time.time()
                    
                    # Configure early stopping for final model evaluation
                    final_model_params = self.best_params.copy()
                    final_model_params['early_stopping_rounds'] = self.early_stopping_rounds
                    final_model_params['eval_metric'] = 'rmse'
                    
                    best_model = XGBRegressor(**final_model_params)
                    evaluation_results = self.model_evaluator.evaluate_model(best_model, self.X, self.y)
                    eval_duration = time.time() - eval_start_time
                    
                    logger.info(f"Final evaluation results: {evaluation_results}")
                    logger.info(f"Final evaluation duration: {eval_duration:.2f} seconds")
                    
                    # Logger le résumé final dans Cosmos DB
                    if self.cosmos_logger:
                        try:
                            final_summary = {
                                "summary_type": "study_completion",
                                "model_type": "xgboost",
                                "total_trials": self.n_trials,
                                "study_duration_seconds": study_duration,
                                "best_rmse": study.best_value,
                                "best_params": self.best_params,
                                "final_evaluation": evaluation_results,
                                "final_eval_duration_seconds": eval_duration,
                                "n_features_used": len(self.X.columns),
                                "feature_selection_method": self.feature_selection_method,
                                "cv_strategy": f"KFold(n_splits={self.n_splits})",
                                "early_stopping_rounds": self.early_stopping_rounds,
                                "eval_metric_used": "rmse",
                                "timestamp": time.time()
                            }
                            self.cosmos_logger.log_trial(final_summary)
                        except Exception as log_error:
                            logger.warning(f"Failed to log final summary to Cosmos DB: {log_error}")
                    
                    # Sauvegarder le modèle si le saver est disponible
                    if self.model_saver:
                        model_path = self.model_saver.save_model(best_model, "xgboost", self.best_params)
                        logger.info(f"Model saved to: {model_path}")
                        
                except Exception as eval_error:
                    logger.error(f"Error during final evaluation: {eval_error}")
            
            return self.best_params
            
        except Exception as e:
            study_duration = time.time() - study_start_time
            logger.error(f"Error during study optimization after {study_duration:.2f}s: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None


# Test du tuner si exécuté directement
if __name__ == "__main__":
    # Test avec des données factices
    print("Testing XGBoost Tuner...")
    
    # Créer des données factices
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                     columns=[f'feature_{i}' for i in range(n_features)])
    y = pd.Series(np.random.randn(n_samples))
    
    print(f"Created test data: X shape {X.shape}, y shape {y.shape}")
    
    # Initialiser le tuner
    tuner = XGBoostTuner(
        X=X, 
        y=y, 
        n_trials=3,  # Peu d'essais pour le test
        use_gpu=False,  # CPU pour la stabilité
        optuna_params=None  # Utiliser les paramètres par défaut
    )
    
    # Lancer l'optimisation
    best_params = tuner.run_study()
    
    if best_params:
        print(f"✅ Test successful! Best parameters: {best_params}")
    else:
        print("❌ Test failed!")
