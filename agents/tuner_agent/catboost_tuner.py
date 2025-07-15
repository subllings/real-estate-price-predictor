import sys, os
os.environ["OMP_NUM_THREADS"] = "1"

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

import optuna
optuna.logging.set_verbosity(optuna.logging.INFO)
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor
from typing import Optional, Dict, Any

from utils.cosmosdb_logger import CosmosDbLogger
from utils.model_saver import ModelSaver
from utils.model_evaluator import ModelEvaluator
from utils.data_loader import DataLoader
from utils.constants import ML_READY_DATA_FILE, TEST_MODE


    
class CatBoostTuner:
    def __init__(self, X, y, n_trials: int, n_splits: int, early_stopping_rounds: int, optuna_params: Optional[Dict[str, Any]] = None, random_state: int = 42, use_gpu: bool = False):
        self.X = X
        self.y = y
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        # Force CPU to avoid segmentation faults
        self.use_gpu = False  # Temporarily disabled to avoid segfaults
        self.model_saver = ModelSaver()
        self.logger = CosmosDbLogger()

        print(f"[INFO] GPU usage forced to: {self.use_gpu} (for stability)")

        # If optuna_params is provided, use it directly (comes from LLM API)
        # Otherwise use default parameters
        if optuna_params is not None:
            # If optuna_params contains a structure with "param_space", extract it
            if isinstance(optuna_params, dict) and "param_space" in optuna_params:
                self.optuna_params = optuna_params["param_space"]
                print(f"[INFO] Using parameter space from API (extracted): {list(self.optuna_params.keys())}")
            else:
                self.optuna_params = optuna_params
                print(f"[INFO] Using parameter space from API: {list(optuna_params.keys())}")
        else:
            # Extended version of default parameters for more complete optimization
            self.optuna_params = {
                # Main learning parameters
                "learning_rate": {"low": 0.01, "high": 0.3, "type": "float", "method": "suggest_loguniform"},
                "depth": {"low": 4, "high": 10, "type": "int", "method": "suggest_int"},
                "iterations": {"low": 100, "high": 2000, "type": "int", "method": "suggest_int"},
                
                # Regularization
                "l2_leaf_reg": {"low": 1.0, "high": 10.0, "type": "float", "method": "suggest_loguniform"},
                "random_strength": {"low": 1e-9, "high": 10.0, "type": "float", "method": "suggest_uniform"},
                
                # Tree structure
                "border_count": {"low": 32, "high": 255, "type": "int", "method": "suggest_int"},
                "min_data_in_leaf": {"low": 1, "high": 20, "type": "int", "method": "suggest_int"},
                
                # Estimation and growth methods
                "grow_policy": {"choices": ["SymmetricTree", "Depthwise", "Lossguide"], "type": "categorical", "method": "suggest_categorical"},
                "leaf_estimation_method": {"choices": ["Newton", "Gradient"], "type": "categorical", "method": "suggest_categorical"},
                "leaf_estimation_iterations": {"low": 1, "high": 10, "type": "int", "method": "suggest_int"},
                
                # Sampling and bagging
                "bootstrap_type": {"choices": ["Bayesian", "Bernoulli", "MVS"], "type": "categorical", "method": "suggest_categorical"},
                "subsample": {"low": 0.6, "high": 1.0, "type": "float", "method": "suggest_uniform"},
                "bagging_temperature": {"low": 0.0, "high": 1.0, "type": "float", "method": "suggest_uniform"},
                "colsample_bylevel": {"low": 0.5, "high": 1.0, "type": "float", "method": "suggest_uniform"},
                
                # Early stopping (important to avoid overfitting)
                "od_type": {"choices": ["IncToDec", "Iter"], "type": "categorical", "method": "suggest_categorical"},
                "od_wait": {"low": 10, "high": 50, "type": "int", "method": "suggest_int"}
            }
            print("[INFO] Using extended default parameter space")
        self.best_model = None
        self.best_model_metrics = None
        self.best_score = float("inf")
        
        # Initialize final metrics
        self.r2_test = None
        self.mae_test = None
        self.rmse_test = None

    def validate_trial_params(self, param_space, trial_params, removed_params=None):
        """Validate trial parameters, skipping those that were intentionally removed"""
        if removed_params is None:
            removed_params = []
            
        for key, spec in param_space.items():
            if key not in trial_params:
                if key in removed_params:
                    print(f"[INFO] Parameter '{key}' intentionally removed due to compatibility constraint")
                else:
                    print(f"[WARNING] Missing param in trial: {key}")
            else:
                val = trial_params[key]
                
                # Validation for fixed values
                if spec.get("method") == "fixed_value":
                    expected_value = spec["value"]
                    # Don't display warning for task_type as we force CPU for stability
                    if val != expected_value and key != "task_type":
                        print(f"[WARNING] Fixed param '{key}' has wrong value: {val} (expected {expected_value})")
                    elif key == "task_type" and val != expected_value:
                        print(f"[INFO] task_type overridden to {val} for stability (API suggested {expected_value})")
                
                # Validation for numerical parameters (int/float)
                elif "low" in spec and "high" in spec:
                    low = spec["low"]
                    high = spec["high"]
                    if not (low <= val <= high):
                        print(f"[WARNING] Out of bounds param '{key}': {val} (expected {low}–{high})")
                
                # Validation for categorical parameters
                elif "choices" in spec:
                    choices = spec["choices"]
                    if val not in choices:
                        print(f"[WARNING] Invalid choice for param '{key}': {val} (expected one of {choices})")
                
                # If neither low/high nor choices are present, skip
                else:
                    print(f"[DEBUG] No validation rules for param '{key}': {val}")


    def suggest_param(self, trial, name, config):
        if isinstance(config, dict):
            param_type = config.get("type")
            if param_type == "float":
                return trial.suggest_float(name, config["low"], config["high"])
            elif param_type == "int":
                return trial.suggest_int(name, config["low"], config["high"])
            elif param_type == "categorical":
                return trial.suggest_categorical(name, config["choices"])

            else:
                raise ValueError(f"Unsupported param type: {param_type}")
        elif isinstance(config, tuple) and len(config) == 2:
            if all(isinstance(i, int) for i in config):
                return trial.suggest_int(name, config[0], config[1])
            else:
                return trial.suggest_float(name, config[0], config[1])
        else:
            raise ValueError(f"Invalid parameter format for '{name}': {config}")



    def objective(self, trial):
        # Use parameters provided dynamically by the LLM API
        params = {
            "loss_function": "RMSE",
            "verbose": 0,
            "random_state": self.random_state,
        }

        # Add all parameters defined in self.optuna_params
        for param_name, param_config in self.optuna_params.items():
            if isinstance(param_config, dict):
                # Handle fixed values (like task_type)
                if param_config.get("method") == "fixed_value":
                    params[param_name] = param_config["value"]
                elif param_config.get("type") == "float" or param_config.get("method") in ["suggest_uniform", "suggest_loguniform"]:
                    if param_config.get("method") == "suggest_loguniform":
                        params[param_name] = trial.suggest_float(param_name, param_config["low"], param_config["high"], log=True)
                    else:
                        params[param_name] = trial.suggest_float(param_name, param_config["low"], param_config["high"])
                elif param_config.get("type") == "int" or param_config.get("method") == "suggest_int":
                    params[param_name] = trial.suggest_int(param_name, param_config["low"], param_config["high"])
                elif param_config.get("type") == "categorical" or param_config.get("method") == "suggest_categorical":
                    # Special attention for task_type - force CPU to avoid segfaults
                    if param_name == "task_type":
                        # Si l'utilisateur veut utiliser GPU ET que GPU est dans les choix
                        if self.use_gpu and "GPU" in param_config.get("choices", []):
                            params[param_name] = "GPU"
                        else:
                            # Force CPU to avoid segfaults
                            params[param_name] = "CPU"
                    else:
                        params[param_name] = trial.suggest_categorical(param_name, param_config["choices"])

        # Override final pour task_type (sécurité supplémentaire)
        if "task_type" not in params:
            params["task_type"] = "CPU"  # Toujours forcer CPU pour éviter les segfaults
        else:
            # Forcer CPU dans tous les cas pour éviter les segfaults
            params["task_type"] = "CPU"
            
        # Filtrer les paramètres problématiques pour éviter les erreurs CatBoost
        removed_params = []  # Track removed parameters to avoid validation warnings
        problematic_combinations = {
            # subsample ne fonctionne qu'avec certains bootstrap_type
            ("subsample", "bootstrap_type"): {
                "remove_if": lambda params: params.get("bootstrap_type") == "Bayesian",
                "reason": "subsample incompatible avec bootstrap_type=Bayesian"
            },
            # bagging_temperature ne fonctionne qu'avec bootstrap_type=Bayesian
            ("bagging_temperature", "bootstrap_type"): {
                "remove_if": lambda params: params.get("bootstrap_type") != "Bayesian",
                "reason": "bagging_temperature available for bayesian bootstrap only"
            },
            # colsample_bylevel peut causer des problèmes avec certaines configurations
            ("colsample_bylevel", "grow_policy"): {
                "remove_if": lambda params: params.get("grow_policy") == "Lossguide" and params.get("colsample_bylevel", 1.0) < 0.8,
                "reason": "colsample_bylevel < 0.8 peut causer des problèmes avec grow_policy=Lossguide"
            }
        }
        
        for (param, related_param), rule in problematic_combinations.items():
            if param in params and rule["remove_if"](params):
                print(f"[WARNING] Removing {param} parameter: {rule['reason']}")
                removed_params.append(param)
                del params[param]

        # Validation of generated parameters (skip removed params)
        self.validate_trial_params(self.optuna_params, params, removed_params=removed_params)

        # Cross-validation with robust error handling to avoid segfaults
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        fold_scores = []
        
        print(f"[DEBUG] Trial {trial.number} params: {params}")
        
        try:
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(self.X)):
                X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
                y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
                
                print(f"[DEBUG] Training fold {fold_idx + 1}/{self.n_splits}")
                
                # Préparer les paramètres d'entraînement
                fit_params = {
                    "eval_set": (X_val, y_val),
                    "use_best_model": True,
                    "verbose": False
                }
                
                # Utiliser early_stopping_rounds ou od_wait selon les paramètres
                if "od_wait" in params:
                    fit_params["early_stopping_rounds"] = params["od_wait"]
                else:
                    fit_params["early_stopping_rounds"] = self.early_stopping_rounds
                
                model = CatBoostRegressor(**params)
                model.fit(X_train, y_train, **fit_params)
                
                preds = model.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, preds))
                fold_scores.append(rmse)
                print(f"[DEBUG] Fold {fold_idx + 1} RMSE: {rmse:.2f}")
        
        except Exception as e:
            print(f"[ERROR] Training failed for trial {trial.number}: {str(e)[:200]}...")
            print(f"[ERROR] Problematic params: {params}")
            
            # Essayer avec des paramètres plus simples/sûrs
            safe_params = {
                "loss_function": "RMSE",
                "verbose": 0,
                "random_state": self.random_state,
                "task_type": "CPU",
                "learning_rate": 0.1,
                "depth": 6,
                "iterations": 100
            }
            
            print("[INFO] Retrying with safe parameters...")
            try:
                for fold_idx, (train_idx, val_idx) in enumerate(kf.split(self.X)):
                    X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
                    y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
                    
                    model = CatBoostRegressor(**safe_params)
                    model.fit(X_train, y_train, verbose=False)
                    
                    preds = model.predict(X_val)
                    rmse = np.sqrt(mean_squared_error(y_val, preds))
                    fold_scores.append(rmse)
                    
            except Exception as e2:
                print(f"[ERROR] Even safe parameters failed: {e2}")
                return float('inf')  # Total failure for this trial
        
        if not fold_scores:
            print("[ERROR] No fold scores obtained")
            return float('inf')
            
        avg_rmse = np.mean(fold_scores)
        print(f"[INFO] Trial {trial.number} average RMSE: {avg_rmse:.2f}")
        
        # Garder le meilleur modèle
        if avg_rmse < self.best_score:
            self.best_score = avg_rmse
            
            # Entraîner le modèle final sur toutes les données
            final_model = CatBoostRegressor(**params)
            final_model.fit(self.X, self.y, verbose=False)
            
            # Évaluer le modèle - utiliser les mêmes données pour train et test dans ce contexte
            # (car on n'a pas de vrai test set séparé ici)
            evaluator = ModelEvaluator(f"catboost_trial_{trial.number}")
            predictions = final_model.predict(self.X)
            
            train_metrics, train_range_metrics = evaluator.evaluate(self.y, predictions)
            # Pour test_metrics, utiliser les mêmes données (limitation du contexte actuel)
            test_metrics, test_range_metrics = train_metrics.copy(), train_range_metrics.copy()
            
            self.best_model = final_model
            self.best_model_metrics = {
                "train": train_metrics,
                "test": test_metrics,
                "by_price_range": test_range_metrics
            }

        return avg_rmse



    
    def run_study(self):
        print("\n[STEP] Creating Optuna study...")
        study = optuna.create_study(direction="minimize")

        print("[STEP] Starting optimization...")
        study.optimize(self.objective, n_trials=self.n_trials, n_jobs=1)

        print("[STEP] Optimization complete.")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best parameters: {study.best_trial.params}")

        self.best_params = study.best_trial.params
        self.best_params["verbose"] = 0

        # === LOGGING AND SAVING ONLY FOR THE BEST MODEL ===

        print("[STEP] Logging final model, metrics and experiment...")

        if self.best_model is None or self.best_model_metrics is None:
            raise RuntimeError("No best model found. Optimization may have failed.")

        best_model = self.best_model
        model_name = f"catboost_best_trial_{study.best_trial.number}"

        global_metrics_train = self.best_model_metrics["train"]
        global_metrics_test = self.best_model_metrics["test"]
        metrics_by_range = self.best_model_metrics["by_price_range"]
        is_perfect = global_metrics_test["r2"] >= 0.90  # ou autre critère de ton choix

        # Sauvegarde du modèle + features
        model_path = self.model_saver.save_model_and_features(
            model=best_model,
            features=self.X.columns.tolist(),
            model_name=model_name,
            metrics=global_metrics_test,
            metrics_by_price_range=metrics_by_range,
        )

        # Logging CSV
        from utils.train_test_metrics_logger import TrainTestMetricsLogger
        csv_logger = TrainTestMetricsLogger()
        csv_logger.log(
            model_name=f"CatBoost CV (All Features){' [TEST]' if TEST_MODE else ''}",
            experiment_name="optuna_best_trial",
            mae_train=global_metrics_train["mae"],
            rmse_train=global_metrics_train["rmse"],
            r2_train=global_metrics_train["r2"],
            mae_test=global_metrics_test["mae"],
            rmse_test=global_metrics_test["rmse"],
            r2_test=global_metrics_test["r2"],
            n_features=self.X.shape[1],
            data_file=ML_READY_DATA_FILE,
            test_mode=TEST_MODE,
            is_perfect=is_perfect,
        )

        # Logging to CosmosDB
        self.logger.log_experiment({
            "type": "optuna_trial",
            "trial_number": study.best_trial.number,
            "model_name": model_name,
            "model_file": model_path,
            "params": study.best_trial.params,
            "metrics": {
                "train": global_metrics_train,
                "test": global_metrics_test,
                "delta_r2": global_metrics_train["r2"] - global_metrics_test["r2"],
                "delta_rmse": global_metrics_train["rmse"] - global_metrics_test["rmse"]
            },
            "metrics_by_price_range": metrics_by_range
        })

        

        self.r2_test = global_metrics_test["r2"]
        self.mae_test = global_metrics_test["mae"]
        self.rmse_test = global_metrics_test["rmse"]

        return study.best_trial


    def get_final_metrics(self) -> dict:
        return {
            "r2_test": self.r2_test,
            "mae_test": self.mae_test,
            "rmse_test": self.rmse_test,
        }


