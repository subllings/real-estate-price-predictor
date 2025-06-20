from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from ml_models.base_model import BaseModel

class RFModel(BaseModel):
    """
    Random Forest model for real estate price prediction.
    """

    def build_pipeline(self):
        numeric_features = ["surface", "bedrooms", "bathrooms", "toilets", "postal_code"]
        categorical_features = ["property_type", "town", "condition"]

        preprocessor = ColumnTransformer(transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ])

        self.pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(n_estimators=100, random_state=42))
        ])
# Placeholder file - to be implemented
