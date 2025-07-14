from .catboost_tuner import CatBoostTuner
from .xgboost_tuner import XGBoostTuner
from .llm_tuner_agent import LLMTunerAgent 
from utils.cosmosdb_logger import CosmosDbLogger
from utils.data_loader import DataLoader
from utils.constants import ML_READY_DATA_FILE, TEST_MODE
from utils.model_evaluator import ModelEvaluator


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

    
    def run(self):
        print(f"\n[INFO] Launching tuning for model: {self.model_name}")

        if TEST_MODE:
            print("TEST MODE ACTIVE – Using limited config for debug purposes\n")
        else:
            print("Running in FULL mode\n")

        print("[STEP] Loading parameter space via ChatGPT...")
        agent = LLMTunerAgent(self.model_name)
        search_space = agent.suggest_param_space()

        print("[STEP] Loading training data...")
        X, y = self._load_training_data()

        print("[STEP] Starting tuning session...")
        best_trial = self._tune_model(search_space, X, y)  # récupère le résultat complet du tuning

        # --- Nouvelle partie : évaluer si modèle parfait ---
        evaluator = ModelEvaluator(self.model_name)
        # Supposons que best_trial contient y_true_train, y_pred_train, y_true_test, y_pred_test
        # Tu dois récupérer ces arrays selon ta structure (exemple ci-dessous)
        y_true_train = best_trial["y_true_train"]
        y_pred_train = best_trial["y_pred_train"]
        y_true_test = best_trial["y_true_test"]
        y_pred_test = best_trial["y_pred_test"]

        is_perfect = ModelEvaluator.is_model_perfect(evaluator, y_true_train, y_pred_train, y_true_test, y_pred_test)

        if is_perfect:
            print("[INFO] Perfect model reached, stopping tuning early.")

        print(f"[DONE] Tuning completed for model: {self.model_name}")
        return is_perfect


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
