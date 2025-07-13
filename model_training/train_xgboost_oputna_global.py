import sys, os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import optuna
import pandas as pd
import numpy as np
from uuid import uuid4
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score

from utils.constants import LOG_COSMOS_DB, ERASE_COSMOS_DB, TEST_MODE, ML_READY_DATA_FILE
from utils.model_evaluator import ModelEvaluator
from utils.train_test_metrics_logger import TrainTestMetricsLogger
from utils.model_saver import ModelSaver

if LOG_COSMOS_DB:
    from utils.cosmosdb_logger import CosmosDbLogger

print("Starting XGBoost Optuna tuning...")
if TEST_MODE:
    print("[TEST_MODE ENABLED] → 3 trials / 1000 rows / 50 estimators")
else:
    print("Running in FULL mode")


class XGBoostOptunaTrainer:
    def __init__(self, n_trials=50, model_name="XGBoost + Optuna (All Features)", use_gpu=True):
        self.n_trials = 3 if TEST_MODE else n_trials
        self.model_name = model_name
        self.best_params = None
        self.model = None
        self.use_gpu = use_gpu
        self.run_id = str(uuid4())

    def load_data(self):
        df = pd.read_csv(ML_READY_DATA_FILE)
        if TEST_MODE:
            df = df.head(1000)
        X = df.drop(columns=["price"])
        y = df["price"]
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def objective(self, trial):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            "n_estimators": 50 if TEST_MODE else 1000,
            "tree_method": "gpu_hist" if self.use_gpu else "auto",
            "verbosity": 0
        }

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []

        for train_idx, valid_idx in kf.split(self.X_dev):
            X_train, X_valid = self.X_dev.iloc[train_idx], self.X_dev.iloc[valid_idx]
            y_train, y_valid = self.y_dev.iloc[train_idx], self.y_dev.iloc[valid_idx]

            model = XGBRegressor(**params)
            model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],
                      early_stopping_rounds=20, verbose=False)

            preds = model.predict(X_valid)
            rmse = np.sqrt(mean_squared_error(y_valid, preds))
            scores.append(rmse)

        return np.mean(scores)

    def tune_hyperparameters(self):
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=self.n_trials)
        self.best_params = study.best_params
        self.best_params["n_estimators"] = 50 if TEST_MODE else 1000
        self.best_params["tree_method"] = "gpu_hist" if self.use_gpu else "auto"
        self.best_params["verbosity"] = 0

    def train_final_model(self):
        self.model = XGBRegressor(**self.best_params)
        self.model.fit(self.X_dev, self.y_dev)

    def evaluate_and_log(self):
        y_train_pred = self.model.predict(self.X_dev)
        y_test_pred = self.model.predict(self.X_test)

        evaluator = ModelEvaluator(self.model_name)
        global_metrics, _ = evaluator.evaluate(self.y_test, y_test_pred)
        evaluator.print_evaluation(self.y_test, y_test_pred)

        mae_train = np.mean(np.abs(self.y_dev - y_train_pred))
        rmse_train = np.sqrt(mean_squared_error(self.y_dev, y_train_pred))
        r2_train = r2_score(self.y_dev, y_train_pred)

        logger = TrainTestMetricsLogger()
        logger.log(
            model_name=self.model_name,
            experiment_name=self.model_name + (" [TEST]" if TEST_MODE else ""),
            mae_train=mae_train,
            rmse_train=rmse_train,
            r2_train=r2_train,
            mae_test=global_metrics["mae"],
            rmse_test=global_metrics["rmse"],
            r2_test=global_metrics["r2"],
            data_file=ML_READY_DATA_FILE,
            n_features=self.X_dev.shape[1]
        )
        logger.display_table()

        ModelSaver().save_model_and_features(
            self.model,
            self.X_dev.columns.tolist(),
            "xgboost_optuna_all_features"
        )

        if LOG_COSMOS_DB:
            cosmos_logger = CosmosDbLogger()

            if ERASE_COSMOS_DB:
                print("[ERASE MODE ENABLED] → Deleting previous logs for this model.")
                cosmos_logger.delete_all_runs(self.model_name)

            delta_rmse = global_metrics["rmse"] - rmse_train
            delta_r2 = r2_train - global_metrics["r2"]
            agent_ready = bool(delta_rmse < 5000 and delta_r2 < 0.05)

            cleaned_params = {
                k: (
                    float(v) if isinstance(v, np.floating)
                    else int(v) if isinstance(v, np.integer)
                    else bool(v) if isinstance(v, np.bool_)
                    else v
                )
                for k, v in self.best_params.items()
            }

            metrics_dict = {
                "run_id": str(uuid4()),
                "model_name": self.model_name,
                "experiment_name": self.model_name + (" [TEST]" if TEST_MODE else ""),
                "train_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "mae_train": mae_train,
                "rmse_train": rmse_train,
                "r2_train": r2_train,
                "mae_test": global_metrics["mae"],
                "rmse_test": global_metrics["rmse"],
                "r2_test": global_metrics["r2"],
                "delta_rmse": round(delta_rmse, 2),
                "delta_r2": round(delta_r2, 4),
                "agent_finetuning_ready": agent_ready,
                "data_file": ML_READY_DATA_FILE,
                "n_features": self.X_dev.shape[1],
                **cleaned_params
            }

            cosmos_logger.log_metrics(metrics_dict)

    def run(self):
        self.X_dev, self.X_test, self.y_dev, self.y_test = self.load_data()
        self.tune_hyperparameters()
        self.train_final_model()
        self.evaluate_and_log()


if __name__ == "__main__":
    trainer = XGBoostOptunaTrainer()
    trainer.run()
