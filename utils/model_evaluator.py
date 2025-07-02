import os
import pandas as pd
from IPython.display import display
from datetime import datetime
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import root_mean_squared_error
from utils.experiment_tracker import ExperimentTracker
from utils.model_visualizer import ModelVisualizer
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from IPython.display import display, HTML
import numpy as np



class ModelEvaluator:
    """
    model_evaluator.py

    This module provides the `ModelEvaluator` class for evaluating and comparing regression models.
    It computes key performance metrics (MAE, RMSE, R²), stores them in memory, and exports them to a timestamped CSV file.
    It also provides functionality to compare multiple evaluations and highlight the best model.

    Classes:
        - ModelEvaluator: Evaluate model performance and save/compare results.

    Typical usage:
        >>> evaluator = ModelEvaluator("XGBoost Regressor")
        >>> mae, rmse, r2 = evaluator.evaluate(y_test, y_pred)
        >>> evaluator.save_to_csv()
        >>> df_results = evaluator.compare_models()
        >>> print(df_results)

    Dependencies:
        - pandas
        - datetime
        - os
        - scikit-learn (sklearn.metrics)
    """

    def __init__(self, model_name=None):
        self.model_name = model_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.records = []
        self.output_path = f"results/eval_results_{self.timestamp}.csv"

    @staticmethod
    def root_mean_squared_error(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    def evaluate(self, y_true, y_pred, dataset_type=""):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = self.root_mean_squared_error(y_true, y_pred)  
        r2 = r2_score(y_true, y_pred)
        
        self.records.append({
            "timestamp": self.timestamp,
            "model": self.model_name,
            "dataset_type": dataset_type,  
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "r2": round(r2, 4)
        })
        self._print_results(mae, rmse, r2, dataset_type)
        
        return mae, rmse, r2



    """
    def save_to_csv(self):
        
        Saves the evaluation records to a CSV file.

        Converts the list of records stored in `self.records` into a pandas DataFrame
        and writes it to a CSV file at the location specified by `self.output_path`.
        The CSV file will not include the DataFrame index.

        Prints the path to the saved results upon completion.
        
        df = pd.DataFrame(self.records)
        df.to_csv(self.output_path, index=False)
        print(f"\nResults saved to: {self.output_path}")
        """

    def compare_models(self):
        """
        Compares recorded model evaluation metrics and identifies the best model based on the highest R² score.

        Returns:
          pandas.DataFrame: A DataFrame containing all model records with an additional 'best' column.
                    The row corresponding to the model with the highest R² score is marked with '✓' in the 'best' column.
                    Returns None if there are no records to compare.
        """
        df = pd.DataFrame(self.records)
        if df.empty:
            print("No model records to compare.")
            return None
        best_idx = df["r2"].idxmax()
        df["best"] = ""
        df.loc[best_idx, "best"] = "✓"
        return df

    def _print_results(self, mae, rmse, r2, dataset_type=""):
        """
        Prints the evaluation metrics for the model in a formatted manner.

        Args:
        mae (float): Mean Absolute Error of the model predictions.
        rmse (float): Root Mean Squared Error of the model predictions.
        r2 (float): R-squared (coefficient of determination) of the model predictions.

        Outputs:
        Prints the model name and the provided evaluation metrics (MAE, RMSE, R²) to the console.
        """
        label = self.model_name or "Model Evaluation"
        if dataset_type:
            label += f" ({dataset_type})"
        print(f"Evaluation – {label}")
        print(f"  MAE:  {mae:,.2f} €")
        print(f"  RMSE: {rmse:,.2f} €")
        print(f"  R²:   {r2:.4f}")
        print("-" * 40)







    def display_all_model_summaries_from_df(self, df_all_evals: pd.DataFrame):
        """
        Display evaluation summaries for all models in a given evaluation DataFrame.
        """
        for model_name in df_all_evals['model'].unique():
            print(f"\n--- Evaluation Summary: {model_name} ---")
            df_model = df_all_evals[df_all_evals['model'] == model_name].copy()
            evaluator = ModelEvaluator(model_name)
            evaluator.display_model_summary(df_model)




    def evaluate_and_track_model(self, model, X_test, y_test, y_pred, model_name, experiment_name=None):
        """
        Evaluation pipeline (simplified):
        - evaluates the model
        - logs metrics
        - displays summary
        - runs visual diagnostics (residuals only)

        Args:
            model: Trained model
            X_test (DataFrame): Test features
            y_test (Series): True target values
            y_pred (array-like): Model predictions
            model_name (str): Name of the model
            experiment_name (str, optional): Experiment label
        """

        # Evaluate
        mae, rmse, r2 = self.evaluate(y_test, y_pred)

        # Log
        tracker = ExperimentTracker()
        df_metrics = tracker.log_and_get_evaluations(
            model=model_name,
            experiment=experiment_name or f"Run {self.timestamp}",
            mae=mae,
            rmse=rmse,
            r2=r2,
        )

        # Display summary
        self.display_model_summary(df_metrics)

        # Basic visual diagnostics
        visualizer = ModelVisualizer(model, X_test, y_test, model_name=model_name)
        visualizer.plot_all_diagnostics()
        visualizer.plot_price_range_residuals()











    @staticmethod
    def plot_price_range_residuals_static(y_true, y_pred, model_name="Model"):
        """
        Plots the distribution of residuals across price ranges using a boxplot.
        """
        residuals = y_true - y_pred
        df = pd.DataFrame({
            "price": y_true,
            "residuals": residuals
        })

        bins = [0, 250_000, 500_000, 750_000, 1_000_000, float("inf")]
        labels = ["<250k", "250k–500k", "500k–750k", "750k–1M", ">1M"]
        df["price_range"] = pd.cut(df["price"], bins=bins, labels=labels)

        plt.figure(figsize=(10, 6))
        sns.boxplot(x="price_range", y="residuals", data=df, palette="muted")
        plt.axhline(0, linestyle="--", color="red")
        plt.title(f"Residuals by Price Range – {model_name}")
        plt.xlabel("Price Range (€)")
        plt.ylabel("Residual (€)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()




    @staticmethod
    def plot_price_range_residuals_side_by_side(y_true, y_pred_1, y_pred_2, model_names=("Model 1", "Model 2")):
        """
        Plots side-by-side boxplots of residuals across price ranges for two models.
        """


        residuals_1 = y_true - y_pred_1
        residuals_2 = y_true - y_pred_2

        df1 = pd.DataFrame({
            "Price Range": pd.cut(y_true, bins=[0, 250_000, 500_000, 750_000, 1_000_000, float("inf")],
                                labels=["<250k", "250k–500k", "500k–750k", "750k–1M", ">1M"]),
            "Residuals": residuals_1,
            "Model": model_names[0]
        })

        df2 = pd.DataFrame({
            "Price Range": pd.cut(y_true, bins=[0, 250_000, 500_000, 750_000, 1_000_000, float("inf")],
                                labels=["<250k", "250k–500k", "500k–750k", "750k–1M", ">1M"]),
            "Residuals": residuals_2,
            "Model": model_names[1]
        })

        df = pd.concat([df1, df2])

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x="Price Range", y="Residuals", hue="Model", palette="Set2")
        plt.axhline(0, linestyle="--", color="red")
        plt.title("Residuals by Price Range (Model Comparison)")
        plt.xlabel("Price Range (€)")
        plt.ylabel("Residual (€)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    @staticmethod
    def plot_shap_comparison_beeswarm(model_all, x_all, model_top, x_top):
        """
        Display side-by-side SHAP beeswarm summary plots for two models.
        """
        # Create SHAP explainers
        explainer_all = shap.Explainer(model_all, x_all)
        explainer_top = shap.Explainer(model_top, x_top)

        # Compute SHAP values
        shap_values_all = explainer_all(x_all)
        shap_values_top = explainer_top(x_top)

        # Plot side-by-side beeswarm summary
        plt.figure(figsize=(18, 7))

        plt.subplot(1, 2, 1)
        shap.plots.beeswarm(shap_values_all, max_display=15, show=False)
        plt.title("SHAP Summary – All Features", fontsize=13)

        plt.subplot(1, 2, 2)
        shap.plots.beeswarm(shap_values_top, max_display=15, show=False)
        plt.title("SHAP Summary – Top RF Features", fontsize=13)

        plt.tight_layout()
        plt.show()



    def enrich_model_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Marqueur de meilleur modèle
        best_idx = df["r2"].idxmax()
        df["best"] = ""
        df.loc[best_idx, "best"] = "✓"

        # Type de modèle
        def get_model_type(name):
            if "Linear" in name:
                return "Linear"
            elif "Random Forest" in name:
                return "Tree"
            elif any(boost in name for boost in ["XGBoost", "LightGBM", "CatBoost"]):
                return "Boosting"
            elif "Stacked" in name:
                return "Ensemble"
            else:
                return "Other"

        df["type"] = df["model"].apply(get_model_type)
        df["rank_r2"] = df["r2"].rank(method="min", ascending=False).astype(int)
        df["r2"] = df["r2"].round(4)

        # Format euros
        df["mae"] = df["mae"].apply(lambda x: f"{x:,.2f} €".replace(",", " "))
        df["rmse"] = df["rmse"].apply(lambda x: f"{x:,.2f} €".replace(",", " "))

        # Ratio rmse / mae
        def parse_euro(value):
            return float(value.replace(" ", "").replace(" €", ""))

        df["rmse/mae"] = df.apply(
            lambda row: round(parse_euro(row["rmse"]) / parse_euro(row["mae"]), 2),
            axis=1
        )

        # Move 'best' column to the end
        best_col = df.pop("best")
        df["best"] = best_col

        return df




    def display_model_summary(self, df: pd.DataFrame):
        if df.empty:
            print("⚠️ No model evaluation records found.")
            return

        # Clean + enrich
        df = df.drop_duplicates().copy()
        best_idx = df["r2"].idxmax()
        df["best"] = ""
        df.loc[best_idx, "best"] = "✓"

        def get_model_type(name):
            if "Linear" in name:
                return "Linear"
            elif "Random Forest" in name:
                return "Tree"
            elif any(boost in name for boost in ["XGBoost", "LightGBM", "CatBoost"]):
                return "Boosting"
            elif "Stacked" in name:
                return "Ensemble"
            else:
                return "Other"

        df["type"] = df["model"].apply(get_model_type)
        df["rank_r2"] = df["r2"].rank(method="min", ascending=False).astype(int)

        # Format € values and ratios
        df["mae"] = df["mae"].apply(lambda x: f"{x:,.2f} €".replace(",", " "))
        df["rmse"] = df["rmse"].apply(lambda x: f"{x:,.2f} €".replace(",", " "))
        df["r2"] = df["r2"].round(4)

        def parse_euro(val):
            return float(val.replace(" ", "").replace(" €", ""))

        df["rmse/mae"] = df.apply(lambda row: round(parse_euro(row["rmse"]) / parse_euro(row["mae"]), 2), axis=1)

        # Reorder: move 'best' to last column
        cols = [col for col in df.columns if col != "best"] + ["best"]
        df = df[cols]

        # Highlight top 3 by rank_r2
        def highlight_top_3(row):
            if row["rank_r2"] == 1:
                return ['background-color: lightgreen'] * len(row)
            elif row["rank_r2"] == 2:
                return ['background-color: #d0f0c0'] * len(row)
            elif row["rank_r2"] == 3:
                return ['background-color: #e6f5d0'] * len(row)
            return [''] * len(row)

        print("=== Model Evaluation Summary ===")
        styled = df.style.apply(highlight_top_3, axis=1)
        display(HTML(styled.to_html()))

        best_model_name = df.loc[best_idx, "model"]
        print(f"\n👉 Best model based on R²: {best_model_name} ✓")
