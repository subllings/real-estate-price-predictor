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


class XGBoostTuner:
    def __init__(self, X, y, n_trials=50, n_splits=5, early_stopping_rounds=20, use_gpu=True, optuna_params=None, random_state=42):
        self.X = X
        self.y = y
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.early_stopping_rounds = early_stopping_rounds
        self.use_gpu = use_gpu
        self.optuna_params = optuna_params or {}
        self.random_state = random_state
        self.best_params = None

    def objective(self, trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", *self.optuna_params.get("learning_rate", (0.01, 0.3))),
            "max_depth": trial.suggest_int("max_depth", *self.optuna_params.get("max_depth", (3, 10))),
            "min_child_weight": trial.suggest_float("min_child_weight", *self.optuna_params.get("min_child_weight", (1, 10))),
            "subsample": trial.suggest_float("subsample", *self.optuna_params.get("subsample", (0.5, 1.0))),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *self.optuna_params.get("colsample_bytree", (0.5, 1.0))),
            "gamma": trial.suggest_float("gamma", *self.optuna_params.get("gamma", (0, 5))),
            "reg_alpha": trial.suggest_float("reg_alpha", *self.optuna_params.get("reg_alpha", (0.0, 1.0))),
            "reg_lambda": trial.suggest_float("reg_lambda", *self.optuna_params.get("reg_lambda", (0.0, 1.0))),
            "n_estimators": 1000,
            "tree_method": "gpu_hist" if self.use_gpu else "auto",
            "verbosity": 0
        }

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        scores = []

        for train_idx, valid_idx in kf.split(self.X):
            X_train, X_valid = self.X.iloc[train_idx], self.X.iloc[valid_idx]
            y_train, y_valid = self.y.iloc[train_idx], self.y.iloc[valid_idx]

            model = XGBRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose=False
            )

            preds = model.predict(X_valid)
            rmse = np.sqrt(mean_squared_error(y_valid, preds))
            scores.append(rmse)

        return np.mean(scores)

    def run_study(self):
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=self.n_trials)
        self.best_params = study.best_params
        self.best_params["n_estimators"] = 1000
        self.best_params["tree_method"] = "gpu_hist" if self.use_gpu else "auto"
        self.best_params["verbosity"] = 0
        return self.best_params
