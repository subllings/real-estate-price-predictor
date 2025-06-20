from abc import ABC, abstractmethod
from sklearn.pipeline import Pipeline
import joblib
import os

class BaseModel(ABC):
    """
    Abstract base class for all real estate price prediction models.
    """

    def __init__(self, name: str = "model"):
        self.name = name
        self.pipeline = None  # Will be defined in build_pipeline()

    @abstractmethod
    def build_pipeline(self):
        """
        Each model must implement its own pipeline.
        Should assign self.pipeline with a sklearn Pipeline object.
        """
        pass

    def train(self, X, y):
        """
        Build the pipeline and fit the model.
        """
        self.build_pipeline()
        self.pipeline.fit(X, y)

    def predict(self, X):
        """
        Make predictions using the trained pipeline.
        """
        if not self.pipeline:
            raise ValueError("Pipeline is not built. Call train() first.")
        return self.pipeline.predict(X)

    def export(self, filepath: str):
        """
        Export the pipeline to a .pkl file.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.pipeline, filepath)
        print(f">>> Model exported to: {filepath}")

    def load(self, filepath: str):
        """
        Load a pipeline from a .pkl file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No model found at: {filepath}")
        self.pipeline = joblib.load(filepath)
        print(f">>> Model loaded from: {filepath}")
