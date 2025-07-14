# tuner_agent/optuna_param_loader.py

class OptunaParamLoader:
    def __init__(self, model_name: str):
        self.model_name = model_name.lower()

    def get_param_space(self) -> dict:
        if self.model_name == "catboost":
            return self._catboost_params()
        elif self.model_name == "xgboost":
            return self._xgboost_params()
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    def _catboost_params(self) -> dict:
        return {
            "learning_rate": (0.01, 0.3),
            "depth": (4, 10),
            "l2_leaf_reg": (1, 10),
            "bagging_temperature": (0.0, 1.0),
            "border_count": (32, 255),
            "random_strength": (0.1, 10.0)
        }

    def _xgboost_params(self) -> dict:
        return {
            "learning_rate": (0.01, 0.3),
            "max_depth": (3, 12),
            "min_child_weight": (1, 10),
            "subsample": (0.5, 1.0),
            "colsample_bytree": (0.5, 1.0),
            "gamma": (0, 5),
            "lambda": (0, 5)
        }
