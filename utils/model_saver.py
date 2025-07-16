import sys, os
# Add the project root to the Python path
project_root = os.path.abspath("../..")
sys.path.append(project_root)

import joblib
import json
from datetime import datetime
from utils.constants import TEST_MODE, MODELS_DIR


class ModelSaver:
    """
    A class to save machine learning models, their features, and associated metrics.
    """    
    def __init__(self):
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.suffix = "_TEST" if TEST_MODE else ""

        self.pkl_dir = os.path.join(MODELS_DIR, "pkl")
        self.features_dir = os.path.join(MODELS_DIR, "features")
        self.metrics_dir = os.path.join(MODELS_DIR, "metrics")

        # Clean or ensure dirs
        if os.path.isfile(self.pkl_dir):
            os.remove(self.pkl_dir)
        os.makedirs(self.pkl_dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

    def save_model(self, model, model_name: str):
        filename = f"{model_name}_{self.run_timestamp}{self.suffix}.pkl"
        path = os.path.join(self.pkl_dir, filename)
        joblib.dump(model, path)
        print(f"[✔] Model saved: {filename}")
        return filename

    def save_features(self, feature_list, model_filename: str):
        json_filename = model_filename.replace(".pkl", ".json")
        json_path = os.path.join(self.features_dir, json_filename)
        with open(json_path, "w") as f:
            json.dump(feature_list, f, indent=2)
        print(f"[✔] Features saved: {json_filename}")
        return json_filename

    def save_metrics(self, metrics_dict: dict, model_filename: str):
        """Save MAE, RMSE, R² in a dedicated metrics JSON file."""
        metrics_data = {
            "model": model_filename,
            "mae": metrics_dict.get("mae"),
            "rmse": metrics_dict.get("rmse"),
            "r2": metrics_dict.get("r2"),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        metrics_filename = model_filename.replace(".pkl", "_metrics.json")
        metrics_path = os.path.join(self.metrics_dir, metrics_filename)
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=2)
        print(f"[✔] Metrics saved: {metrics_filename}")
        return metrics_filename

    def save_metrics_by_price_range(self, metrics_by_range: list, model_filename: str):
            """
            Save metrics (MAE, RMSE, R²) grouped by price range into a JSON file.
            """
            data = {
                "model": model_filename,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "price_range_metrics": metrics_by_range
            }
            metrics_range_filename = model_filename.replace(".pkl", "_metrics_by_price_range.json")
            metrics_range_path = os.path.join(self.metrics_dir, metrics_range_filename)
            with open(metrics_range_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"[✔] Price range metrics saved: {metrics_range_filename}")
            return metrics_range_filename


    def save_model_and_features(
        self,
        model,
        features,
        model_name: str,
        metrics: dict = None,
        metrics_by_price_range: list = None
    ):
        model_filename = self.save_model(model, model_name)
        self.save_features(features, model_filename)
        if metrics:
            self.save_metrics(metrics, model_filename)
        if metrics_by_price_range:
            self.save_metrics_by_price_range(metrics_by_price_range, model_filename)
        return model_filename

