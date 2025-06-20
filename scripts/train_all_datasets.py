import os
import pandas as pd
import joblib
from sklearn.metrics import r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_models.model_factory import ModelFactory
from utils.column_mapper import load_column_mapping, standardize_columns



class DatasetTrainer:
    def __init__(self, data_dir="data", model_dir="local_models", target="price", model_types=["rf", "lr", "dgbm"]):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.target = target
        self.model_types = model_types
        self.mapping_dict = load_column_mapping()


    @staticmethod
    def clean_dataframe(df: pd.DataFrame, target: str) -> pd.DataFrame:
        """
        Cleans the dataframe before training:
        - Cleans price and surface columns (removes currency, spaces, symbols)
        - Converts numerical columns to float
        - Strips text from categorical columns
        - Handles known object columns that must be numeric
        """

        import re

        # Columns to clean as numeric (if present)
        numeric_cols = [
            target, "surface", "terrace_surface", "bedroom1_surface",
            "bedroom2_surface", "epc_score", "epc_total"
        ]

        for col in numeric_cols:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace("€", "", regex=False)
                    .str.replace("m²", "", regex=False)
                    .str.replace("%", "", regex=False)
                    .str.replace("\u202f", "", regex=False)
                    .str.replace("\xa0", "", regex=False)
                    .str.replace(",", ".", regex=False)
                    .str.replace(r"[^\d.]", "", regex=True)
                )
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Optional: strip whitespace from text columns
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].astype(str).str.strip()

        return df




    def train_all(self):
        for filename in os.listdir(self.data_dir):
            if not filename.endswith(".csv"):
                continue

            dataset_path = os.path.join(self.data_dir, filename)
            dataset_name = os.path.splitext(filename)[0]

            try:
                df = pd.read_csv(dataset_path)
                df = standardize_columns(df, self.mapping_dict)

                # Clean all known problematic columns
                df = DatasetTrainer.clean_dataframe(df, self.target)

                if self.target not in df.columns:
                    print(f"[SKIPPED] Target '{self.target}' not found in {filename}")
                    continue


                # Columns to exclude from training
                exclude_columns = ["url", "address", "epc_valid_until", "epc_score"]

                # Split features and target
                X = df.drop(columns=[self.target])
                y = df[self.target]

                # Drop useless columns if present
                X = X.drop(columns=[col for col in exclude_columns if col in X.columns])

                # Encode categorical columns
                for col in X.select_dtypes(include=["object"]).columns:
                    X[col] = X[col].astype("category").cat.codes

                # Drop rows with NaN values in features or target
               # Drop rows with NaN in features
                X = X.dropna()
                y = y.loc[X.index]

                # Drop rows where y is still NaN
                y = y.dropna()
                X = X.loc[y.index]

                # Launch training for each model type
                print(f"[INFO] Training models for dataset: {dataset_name}...")
                self._train_models_for_dataset(dataset_name, X, y)



            except Exception as e:
                print(f"[ERROR] Failed to process {filename}: {str(e)}")



    def _train_models_for_dataset(self, dataset_name, X, y):
        for model_type in self.model_types:
            try:
                model = ModelFactory.create(model_type)

                # Identify column types
                categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
                numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

                # Build preprocessor
                preprocessor = ColumnTransformer(transformers=[
                    ("num", StandardScaler(), numeric_cols),
                    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
                ])

                # Fit and transform the features
                X_preprocessed = preprocessor.fit_transform(X)

                # Fit the model on preprocessed data
                model.fit(X_preprocessed, y)

                # Score on training set
                y_pred = model.predict(X_preprocessed)
                r2 = r2_score(y, y_pred)
                print(f"[SCORE] {model_type} on {dataset_name}: R² = {r2:.3f}")

                # Save model
                model_subdir = os.path.join(self.model_dir, model_type)
                os.makedirs(model_subdir, exist_ok=True)
                model_path = os.path.join(model_subdir, f"{dataset_name}_{model_type}.pkl")
                joblib.dump(model, model_path)

                # Save preprocessor
                preproc_path = os.path.join(model_subdir, f"{dataset_name}_{model_type}_preprocessor.pkl")
                joblib.dump(preprocessor, preproc_path)

                print(f"[OK] {model_type} trained and saved for {dataset_name}")

            except Exception as e:
                print(f"[ERROR] Failed to train {model_type} for {dataset_name}: {str(e)}")

if __name__ == "__main__":
    print(">>> Launching dataset-wide training for all models...")

    trainer = DatasetTrainer(
        data_dir="data",
        model_dir="local_models",
        target="price",
        model_types=["rf", "lr", "dgbm"]
    )

    trainer.train_all()

    print(">>> All models trained successfully for all datasets.")
