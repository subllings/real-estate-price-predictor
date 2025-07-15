# agents/tuner_agent/optuna_param_loader.py

class OptunaParamLoader:
    def __init__(self, model_name):
        self.model_name = model_name.lower()

    def get_param_space(self):
        if self.model_name == "catboost":
            return {
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.3},
                "depth": {"type": "int", "low": 4, "high": 10},
                "l2_leaf_reg": {"type": "float", "low": 1.0, "high": 10.0},
                "bagging_temperature": {"type": "float", "low": 0.0, "high": 1.0},
                "border_count": {"type": "int", "low": 32, "high": 255},
                "random_strength": {"type": "float", "low": 0.0, "high": 10.0},
                "grow_policy": {"type": "categorical", "choices": ["SymmetricTree", "Depthwise", "Lossguide"]},
                "min_data_in_leaf": {"type": "int", "low": 1, "high": 20},
                "leaf_estimation_iterations": {"type": "int", "low": 1, "high": 10},
                "iterations": {"type": "int", "low": 100, "high": 2000}
            }
        elif self.model_name == "xgboost":
            # Define XGBoost parameter space...
            return {}
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
