import joblib
import pandas as pd
import os


class PricePredictor:
    def __init__(self, model_path: str, preprocessor_path: str = None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model = joblib.load(model_path)

        if not hasattr(self.model, "predict"):
            raise TypeError("Loaded object is not a valid model.")

        self.preprocessor = None
        if preprocessor_path:
            if not os.path.exists(preprocessor_path):
                raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")
            self.preprocessor = joblib.load(preprocessor_path)

    def predict(self, input_data: dict) -> float:
        df = pd.DataFrame([input_data])
        if self.preprocessor:
            df = self.preprocessor.transform(df)
        prediction = self.model.predict(df)
        predicted_price = float(prediction[0])
        print(f">>> Predicted price from input: €{predicted_price:,.0f}")
        return predicted_price



if __name__ == "__main__":
    sample = {
        "surface": 120,
        "rooms": 3,
        "province": "west-vlaanderen",
        "postal_code": 8500,
        "property_type": "house",
        "region": "flanders",
        "has_garden": True,
        "has_terrace": False,
        "epc": 200,
        "terrace_surface": 10,
        "bedroom2_surface": 15,
        "bedroom1_surface": 20,
        "toilets": 2,
        "epc_total": 30000,
        "bathrooms": 1,
        "year_built": 2005,
        "floor": 0,
        "page": 1,
        "bedrooms": 3
    }

    models = ["rf", "lr", "dgbm"]
    for model_type in models:
        print(f"\n>>> Predicting with model: {model_type.upper()}")
        model_path = f"local_models/{model_type}/immovlan_real_estate_{model_type}.pkl"
        preproc_path = f"local_models/{model_type}/immovlan_real_estate_{model_type}_preprocessor.pkl"
        predictor = PricePredictor(model_path=model_path, preprocessor_path=preproc_path)
        predicted_price = predictor.predict(sample)
        print(f">>> Predicted price from input ({model_type.upper()}): €{predicted_price:,.0f}")