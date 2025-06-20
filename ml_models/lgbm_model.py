from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from ml_models.base_model import BaseModel

class LGBMModel(BaseModel):
    """
    LightGBM model for real estate price prediction.
    """

    def build_pipeline(self):
        """
        Build a pipeline with optional preprocessing and the LightGBM model.
        """
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LGBMRegressor(n_estimators=100, random_state=42))
        ])
