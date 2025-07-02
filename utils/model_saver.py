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
    A class to save machine learning models and their features.
    
    Example usage: 
      saver = ModelSaver()
      saver.save_model_and_features(model_all, list(X_reduced.columns), "catboost_optuna_all")
      saver.save_model_and_features(model_top, top_features, "catboost_optuna_top30")

    """    
    def __init__(self):
        # Generate a single run-wide timestamp
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.suffix = "_TEST" if TEST_MODE else ""

        # Define directories
        self.pkl_dir = os.path.join(MODELS_DIR, "pkl")
        self.features_dir = os.path.join(MODELS_DIR, "features")

        # Ensure directories exist
        if os.path.isfile(self.pkl_dir):
            os.remove(self.pkl_dir)
        os.makedirs(self.pkl_dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)

    def save_model(self, model, model_name: str):
        """Save a model to disk with timestamp and suffix."""
        filename = f"{model_name}_{self.run_timestamp}{self.suffix}.pkl"
        path = os.path.join(self.pkl_dir, filename)
        joblib.dump(model, path)
        print(f"[✔] Model saved: {filename}")
        return filename

    def save_features(self, feature_list, model_filename: str):
        """Save the list of features to a JSON file."""
        json_filename = model_filename.replace(".pkl", ".json")
        json_path = os.path.join(self.features_dir, json_filename)
        with open(json_path, "w") as f:
            json.dump(feature_list, f, indent=2)
        print(f"[✔] Features saved: {json_filename}")
        return json_filename

    def save_model_and_features(self, model, features, model_name: str):
        """Save both model and feature list."""
        model_filename = self.save_model(model, model_name)
        self.save_features(features, model_filename)
        return model_filename  # Optionally return full filename if needed elsewhere


