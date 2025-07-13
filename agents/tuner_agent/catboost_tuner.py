import sys, os
project_root = os.path.abspath("../..")
sys.path.append(project_root)

import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor

class CatBoostTuner:
    def __init__(self, X, y, n_trials=50, n_splits=5, early_stopping_rounds=20, optuna_params=None, random_state=42):
        self.X = X
        self.y = y
        self.n_trials = n_trials
        self.n_splits = n_splits
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.best_params = None
        self.optuna_params = optuna_params or {}

    def objective(self, trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", *self.optuna_params.get("learning_rate", (0.01, 0.3))),
            "depth": trial.suggest_int("depth", *self.optuna_params.get("depth", (4, 10))),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", *self.optuna_params.get("l2_leaf_reg", (1.0, 10.0))),
            "bagging_temperature": trial.suggest_float("bagging_temperature", *self.optuna_params.get("bagging_temperature", (0.0, 1.0))),
            "border_count": trial.suggest_int("border_count", *self.optuna_params.get("border_count", (32, 255))),
            "random_strength": trial.suggest_float("random_strength", *self.optuna_params.get("random_strength", (1e-9, 10.0))),
            "verbose": 0
        }

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        scores = []

        for train_idx, valid_idx in kf.split(self.X):
            X_train, X_valid = self.X.iloc[train_idx], self.X.iloc[valid_idx]
            y_train, y_valid = self.y.iloc[train_idx], self.y.iloc[valid_idx]

            model = CatBoostRegressor(**params)
            model.fit(X_train, y_train, eval_set=(X_valid, y_valid),
                      early_stopping_rounds=self.early_stopping_rounds, verbose=0)
            
            preds = model.predict(X_valid)
            rmse = np.sqrt(mean_squared_error(y_valid, preds))
            scores.append(rmse)

        return np.mean(scores)

    def run_study(self):
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=self.n_trials)
        self.best_params = study.best_params
        self.best_params["verbose"] = 0
        return self.best_params
