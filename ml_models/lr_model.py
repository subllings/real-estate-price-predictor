
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from ml_models.base_model import BaseModel

class LRModel(BaseModel):
    """
    Linear Regression model for real estate price prediction.
    """

    def build_pipeline(self):
        """
        Build a pipeline with preprocessing and Linear Regression.
        """
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ])
