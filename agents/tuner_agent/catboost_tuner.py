import sys, os
os.environ["OMP_NUM_THREADS"] = "1"

from fastapi import params

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
    def __init__(
        self,
        X,
        y,
        n_trials: int,
        n_splits: int,
        early_stopping_rounds: int,
        optuna_params: Optional[Dict[str, Any]] = None,
        random_state: int = 42,
        use_gpu: bool = False,
    ):
        self.X = X
        self.y = y
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.use_gpu = use_gpu  
        self.model_saver = ModelSaver()
        self.logger = CosmosDbLogger()

        self.optuna_params = optuna_params or {
            "learning_rate": (0.01, 0.3),
            "depth": (4, 10),
            "l2_leaf_reg": (1.0, 10.0),
            "bagging_temperature": (0.0, 1.0),
            "border_count": (32, 255),
            "random_strength": (1e-9, 10.0)
        }

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

        print(f"\n[TRIAL {trial.number}] Started", flush=True)

        # Suggest bootstrap_type first (needed for conditional logic)
        bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"])

        # Build params dictionary
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
            "grow_policy": trial.suggest_categorical("grow_policy", ["SymmetricTree", "Depthwise", "Lossguide"]),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 20),
            "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", 1, 10),
            "leaf_estimation_method": trial.suggest_categorical("leaf_estimation_method", ["Newton", "Gradient"]),
            "bootstrap_type": bootstrap_type,
            "verbose": 1,
        }


        params["thread_count"] = 1  # Use single thread for reproducibility  



        # Conditional parameter only for Bayesian bootstrap
        if bootstrap_type == "Bayesian":
            params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0.0, 1.0)

        # Use GPU if enabled, fallback to CPU if error occurs
        if self.use_gpu:
            params["task_type"] = "GPU"
        else:
            params["task_type"] = "CPU"
        # Force CPU pour test uniquement (commenter/décommenter selon besoin)
        params["task_type"] = "CPU"
        print(f"Training with {params['task_type']}")

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        scores, models, evals = [], [], []

        try:
            # Train and validate with KFold using GPU or CPU as set
            for train_idx, valid_idx in kf.split(self.X):
                X_train, X_valid = self.X.iloc[train_idx], self.X.iloc[valid_idx]
                y_train, y_valid = self.y.iloc[train_idx], self.y.iloc[valid_idx]


                model = CatBoostRegressor(**params)
                model.fit(
                    X_train,
                    y_train,
                    eval_set=(X_valid, y_valid),
                    early_stopping_rounds=self.early_stopping_rounds,
                    verbose=100,
                )

                preds = model.predict(X_valid)
                rmse = np.sqrt(mean_squared_error(y_valid, preds))
                scores.append(rmse)
                models.append(model)
                evals.append((X_valid, y_valid, preds))

        except Exception as e:
            # If GPU training fails, retry with CPU fallback
            if self.use_gpu:
                print(f"[WARNING] GPU training failed with error: {e}. Falling back to CPU.")
                params["task_type"] = "CPU"
                scores.clear()
                models.clear()
                evals.clear()
                for train_idx, valid_idx in kf.split(self.X):
                    X_train, X_valid = self.X.iloc[train_idx], self.X.iloc[valid_idx]
                    y_train, y_valid = self.y.iloc[train_idx], self.y.iloc[valid_idx]

                    model = CatBoostRegressor(**params)
                    model.fit(
                        X_train,
                        y_train,
                        eval_set=(X_valid, y_valid),
                        early_stopping_rounds=self.early_stopping_rounds,
                        verbose=0,
                    )

                    preds = model.predict(X_valid)
                    rmse = np.sqrt(mean_squared_error(y_valid, preds))
                    scores.append(rmse)
                    models.append(model)
                    evals.append((X_valid, y_valid, preds))
            else:
                raise e  # Raise if CPU mode also fails

        # Select best fold based on RMSE
        best_fold_idx = int(np.argmin(scores))
        best_model = models[best_fold_idx]
        X_eval, y_eval, y_pred = evals[best_fold_idx]

        # Evaluate on validation fold
        evaluator = ModelEvaluator(model_name=f"CatBoost_Trial_{trial.number}")
        global_metrics_test, metrics_by_range = evaluator.evaluate(
            y_eval, y_pred, bins=[0, 200000, 400000, 600000, 1000000]
        )
        evaluator.print_evaluation(y_eval, y_pred, bins=[0, 200000, 400000, 600000, 1000000])

        # Evaluate on training fold (best fold train part)
        train_indices, _ = list(kf.split(self.X))[best_fold_idx]
        X_train_best = self.X.iloc[train_indices]
        y_train_best = self.y.iloc[train_indices]

        global_metrics_train, _ = evaluator.evaluate(
            y_train_best, best_model.predict(X_train_best), bins=[0, 200000, 400000, 600000, 1000000]
        )

        # Determine if model is considered perfect
        is_perfect = ModelEvaluator.is_model_perfect(
            evaluator, y_train_best, best_model.predict(X_train_best), y_eval, y_pred
        )

        # Save predictions and true values in user attributes for retrieval later
        trial.set_user_attr("y_true_train", y_train_best.tolist())
        trial.set_user_attr("y_pred_train", best_model.predict(X_train_best).tolist())
        trial.set_user_attr("y_true_test", y_eval.tolist())
        trial.set_user_attr("y_pred_test", y_pred.tolist())

        # Save model with timestamp in filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        model_name = f"catboost_trial_{trial.number}_{'_TEST' if TEST_MODE else ''}"

        model_path = self.model_saver.save_model_and_features(
            model=best_model,
            features=self.X.columns.tolist(),
            model_name=model_name,
            metrics=global_metrics_test,
            metrics_by_price_range=metrics_by_range,
        )

        # Log metrics to CSV
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

        # Log experiment to Cosmos DB
        self.logger.log_experiment(
            {
                "type": "optuna_trial",
                "trial_number": trial.number,
                "model_name": model_name,
                "model_file": model_path,
                "params": params,
                "metrics": {
                    "train": global_metrics_train,
                    "test": global_metrics_test,
                    "delta_r2": global_metrics_train["r2"] - global_metrics_test["r2"],
                    "delta_rmse": global_metrics_train["rmse"] - global_metrics_test["rmse"]
                },
                "metrics_by_price_range": metrics_by_range,
                "is_perfect": is_perfect
            }
        )

        self.r2_test = global_metrics_test["r2"]
        self.mae_test = global_metrics_test["mae"]
        self.rmse_test = global_metrics_test["rmse"]

        return np.mean(scores)


    
    def run_study(self):
        print("\n[STEP] Creating Optuna study...")
        study = optuna.create_study(direction="minimize")
        
        print("[STEP] Starting optimization...")
        study.optimize(self.objective, n_trials=self.n_trials, n_jobs=1) 

        print("[STEP] Optimization complete.")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best parameters: {study.best_trial.params}")

        self.best_params = study.best_params
        self.best_params["verbose"] = 0
        return study.best_trial


    def get_final_metrics(self) -> dict:
        return {
            "r2_test": self.r2_test,
            "mae_test": self.mae_test,
            "rmse_test": self.rmse_test,
        }


