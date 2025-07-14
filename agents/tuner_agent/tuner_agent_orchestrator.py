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
        self.n_trials = 3 if TEST_MODE else 50
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
            tuner = CatBoostTuner(X, y, self.n_trials, self.n_splits, self.early_stopping_rounds,
                                search_space, self.random_state, self.use_gpu)
        elif self.model_name == "xgboost":
            tuner = XGBoostTuner(X, y, self.n_trials, self.n_splits, self.early_stopping_rounds,
                                search_space, self.random_state, self.use_gpu)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        # Step 4 – Run optimization
        print("[STEP] Starting optimization...")
        study = optuna.create_study(direction="minimize")
        study.optimize(
            tuner.objective,
            n_trials=self.n_trials,
            gc_after_trial=True
        )

        best_trial = study.best_trial
        print(f"\n✅ Best trial – RMSE: {best_trial.value:.2f}")

        # Step 5 – Evaluate if the model is "perfect"
        final_metrics = tuner.get_final_metrics()
        r2 = final_metrics.get("r2_test", 0)
        mae = final_metrics.get("mae_test", float("inf"))
        rmse = final_metrics.get("rmse_test", float("inf"))

        r2_gap = 0
        is_perfect = self.is_model_perfect(r2=r2, mae=mae, rmse=rmse, r2_previous=self.best_r2_so_far)
        if is_perfect:
            print("🎯 Perfect or significantly improved model found!")
            print(f"Metrics:\n  R²: {r2:.4f}\n  MAE: {mae:.2f}\n  RMSE: {rmse:.2f}")
            if r2_gap is not None:
                print(f"R² improvement (gap) over previous best: {r2_gap:.4f}")

        return best_trial, is_perfect



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


    def is_model_perfect(self, r2, mae, rmse, r2_previous=None):
        """
        Returns True if the model is considered perfect (meets all quality thresholds),
        or if it shows a significant improvement in R² over the previous model.
        Updates self.best_r2_so_far if the current r2 is better.
        """
        # Cas 1 – modèle parfait selon les trois métriques
        if (
            r2 >= PERFECT_R2_THRESHOLD and
            mae <= PERFECT_MAE_THRESHOLD and
            rmse <= PERFECT_RMSE_THRESHOLD
        ):
            # Mise à jour best_r2_so_far si meilleure valeur trouvée
            if r2 > self.best_r2_so_far:
                self.best_r2_so_far = r2
            return True

        # Cas 2 – amélioration significative du R²
        if r2_previous is not None:
            delta_r2 = r2 - r2_previous
            if delta_r2 >= DELTA_R2_THRESHOLD:
                if r2 > self.best_r2_so_far:
                    self.best_r2_so_far = r2
                return True

        # Sinon : ni parfait, ni significativement meilleur
        return False