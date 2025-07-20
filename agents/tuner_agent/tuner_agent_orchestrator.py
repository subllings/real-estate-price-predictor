import logging
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

# Configuration des logs Azure (doit être fait avant les imports Azure)
from utils.configure_logging import configure_azure_logging

from .catboost_tuner import CatBoostTuner
from .xgboost_tuner import XGBoostTuner
from .llm_tuner_agent import LLMTunerAgent 
from utils.cosmosdb_logger import CosmosDbLogger
from utils.data_loader import DataLoader
from utils.constants import ML_READY_DATA_FILE, TEST_MODE
from utils.model_evaluator import ModelEvaluator
from utils.model_saver import ModelSaver
import optuna

class TunerAgentOrchestrator:
    def __init__(self, model_name: str, n_trials: int = None):
        self.model_name = model_name.lower()
        self.logger = CosmosDbLogger()

        # Handle new model naming (remove "+ Optuna" suffix if present)
        if " + optuna" in self.model_name:
            self.model_name = self.model_name.replace(" + optuna", "")
        
        # Map stack_ensemble to actual implementation
        if self.model_name == "stack_ensemble":
            self.model_name = "catboost"  # Default to catboost for now, could be extended

        # Default settings – override if needed
        self.n_trials = n_trials if n_trials is not None else (1 if TEST_MODE else 50)
        self.n_splits = 5  # 5-fold cross-validation sur les données d'entraînement
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

        # Step 2 – Get parameter search space from LLM agent (ChatGPT)
        print("[STEP] Loading parameter space via ChatGPT...")
        llm_agent = LLMTunerAgent(self.model_name)
        search_space = llm_agent.suggest_param_space()
        print("[✔] Parameter space loaded.")

        # Step 3 – Initialize the tuner based on the model type
        print(f"[STEP] Initializing tuner for {self.model_name}...")
        if self.model_name == "catboost":
            tuner = CatBoostTuner(
                X=X, 
                y=y, 
                n_trials=self.n_trials, 
                n_splits=self.n_splits, 
                early_stopping_rounds=self.early_stopping_rounds,
                optuna_params=search_space, 
                random_state=self.random_state,
                use_gpu=self.use_gpu
            )
        elif self.model_name == "xgboost":
            tuner = XGBoostTuner(
                X=X, 
                y=y, 
                n_trials=self.n_trials, 
                n_splits=self.n_splits, 
                early_stopping_rounds=self.early_stopping_rounds,
                use_gpu=self.use_gpu,
                optuna_params=search_space, 
                random_state=self.random_state,
                feature_selection_method="all_features"  # Peut être configuré dynamiquement
            )
        elif self.model_name == "lightgbm":
            # For now, use XGBoost tuner as a fallback (similar gradient boosting)
            print("[INFO] LightGBM tuner not implemented yet, using XGBoost tuner as fallback")
            tuner = XGBoostTuner(
                X=X, 
                y=y, 
                n_trials=self.n_trials, 
                n_splits=self.n_splits, 
                early_stopping_rounds=self.early_stopping_rounds,
                use_gpu=self.use_gpu,
                optuna_params=search_space, 
                random_state=self.random_state,
                feature_selection_method="all_features"
            )
        elif self.model_name == "random_forest":
            # For now, use CatBoost tuner as a fallback 
            print("[INFO] Random Forest tuner not implemented yet, using CatBoost tuner as fallback")
            tuner = CatBoostTuner(
                X=X, 
                y=y, 
                n_trials=self.n_trials, 
                n_splits=self.n_splits, 
                early_stopping_rounds=self.early_stopping_rounds,
                optuna_params=search_space, 
                random_state=self.random_state,
                use_gpu=self.use_gpu
            )
        elif self.model_name == "stack_ensemble":
            # Use CatBoost as the base for stacked ensemble
            print("[INFO] Using CatBoost as base for stacked ensemble")
            tuner = CatBoostTuner(
                X=X, 
                y=y, 
                n_trials=self.n_trials, 
                n_splits=self.n_splits, 
                early_stopping_rounds=self.early_stopping_rounds,
                optuna_params=search_space, 
                random_state=self.random_state,
                use_gpu=self.use_gpu
            )
        else:
            raise ValueError(f"Unsupported model: {self.model_name}. Supported models: catboost, xgboost, lightgbm, random_forest, stack_ensemble")

        # Step 4 – Run optimization
        print("[STEP] Starting optimization...")
        best_params = tuner.run_study()

        if best_params:
            print(f"\n✅ Optimization completed successfully!")
            print(f"Best parameters: {best_params}")
            return best_params, False
        else:
            print("❌ Optimization failed!")
            return None, False


    def _load_training_data(self):
        loader = DataLoader(ML_READY_DATA_FILE)
        df = loader.load_data()
        return loader.split_X_y(df)



