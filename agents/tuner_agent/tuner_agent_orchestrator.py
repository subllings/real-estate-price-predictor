from .catboost_tuner import CatBoostTuner
from .xgboost_tuner import XGBoostTuner
from .llm_tuner_agent import LLMTunerAgent 
from utils.cosmosdb_logger import CosmosDbLogger
from utils.data_loader import DataLoader
from utils.constants import ML_READY_DATA_FILE, TEST_MODE
from utils.model_evaluator import ModelEvaluator
from agents.tuner_agent.optuna_param_loader import OptunaParamLoader
import optuna
from utils.constants import (
    PERFECT_R2_THRESHOLD,
    PERFECT_MAE_THRESHOLD,
    PERFECT_RMSE_THRESHOLD,
    DELTA_R2_THRESHOLD
)

class TunerAgentOrchestrator:
    def __init__(self, model_name: str):
        self.model_name = model_name.lower()
        self.logger = CosmosDbLogger()

        # Default settings – override if needed
        self.n_trials = 1 if TEST_MODE else 50
        self.n_splits = 3
        self.early_stopping_rounds = 20
        self.use_gpu = True
        self.random_state = 42
        self.best_r2_so_far = float("-inf")

    
    def run(self) -> tuple:
        print(f"\n[INFO] Launching tuning for model: {self.model_name}")
        if TEST_MODE:
            print("TEST MODE ACTIVE - Using limited config for debug purposes")

        # Step 1 – Load data
        data_loader = DataLoader(ML_READY_DATA_FILE)
        df = data_loader.load_data()
        X, y = data_loader.split_X_y(df)

        # Step 2 – Load parameter search space via GPT
        print("[STEP] Loading parameter space via ChatGPT...")
        param_loader = OptunaParamLoader(self.model_name)
        search_space = param_loader.get_param_space()
        print("[✔] Parameter space loaded.")

        # Step 3 – Initialize the tuner
        if self.model_name == "catboost":
            tuner = CatBoostTuner(
                X, y, self.n_trials, self.n_splits, self.early_stopping_rounds,
                search_space, self.random_state, self.use_gpu
            )
        elif self.model_name == "xgboost":
            tuner = XGBoostTuner(
                X, y, self.n_trials, self.n_splits, self.early_stopping_rounds,
                search_space, self.random_state, self.use_gpu
            )
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        # Step 4 – Run optimization (avec run_study() qui fait tout)
        print("[STEP] Starting optimization...")
        best_trial = tuner.run_study()

        print(f"\n✅ Best trial – RMSE: {best_trial.value:.2f}")

        # Récupération des métriques finales
        final_metrics = tuner.get_final_metrics()
        r2 = final_metrics.get("r2_test", 0)
        mae = final_metrics.get("mae_test", float("inf"))
        rmse = final_metrics.get("rmse_test", float("inf"))

        print(f"Final metrics:\n  R²: {r2:.4f}\n  MAE: {mae:.2f}\n  RMSE: {rmse:.2f}")

        # Ici on ne fait plus de vérification 'is_perfect' basée sur des attributs absents
        return best_trial, False


    def _tune_model(self, search_space, X, y):
        common_args = {
            "X": X,
            "y": y,
            "n_trials": self.n_trials,
            "n_splits": self.n_splits,
            "early_stopping_rounds": self.early_stopping_rounds,
            "use_gpu": self.use_gpu,
            "optuna_params": search_space,
            "random_state": self.random_state,
        }

        if self.model_name == "xgboost":
            tuner = XGBoostTuner(**common_args)
        elif self.model_name == "catboost":
            tuner = CatBoostTuner(**common_args)
        else:
            raise ValueError(f"[ERROR] Unsupported model: {self.model_name}")

        return tuner.run_study()


    def _load_training_data(self):
        loader = DataLoader(ML_READY_DATA_FILE)
        df = loader.load_data()
        return loader.split_X_y(df)


