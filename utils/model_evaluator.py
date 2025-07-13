import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class ModelEvaluator:
    def __init__(self, model_name):
        # Store the model identifier for printouts
        self.model_name = model_name

    def evaluate(self, y_true, y_pred, bins=None, dataset_type=None):
        """
        Evaluate model performance using global and (optional) price range metrics.
        """
        global_metrics = self._compute_global_metrics(y_true, y_pred)
        range_metrics = self._compute_metrics_by_price_range(y_true, y_pred, bins) if bins else []

        if dataset_type:
            print(f"[{self.model_name}] Evaluation on {dataset_type} set")
            print(f"  MAE:  {global_metrics['mae']:,.2f} €")
            print(f"  RMSE: {global_metrics['rmse']:,.2f} €")
            print(f"  R²:   {global_metrics['r2']:.4f}")
            print("-" * 40)

        return global_metrics, range_metrics




    def print_evaluation(self, y_true, y_pred, bins=None):
        """
        Print global metrics and segmented performance by price range.
        """
        print(f"[DEBUG] Call print_evaluation for {self.model_name}")
        print(f"[DEBUG] y_true: {len(y_true)}, y_pred: {len(y_pred)}")

        global_metrics, range_metrics = self.evaluate(y_true, y_pred, bins)

        print(f"\nEvaluation – {self.model_name}")
        print(f"  MAE:  {global_metrics['mae']:,.2f} €")
        print(f"  RMSE: {global_metrics['rmse']:,.2f} €")
        print(f"  R²:   {global_metrics['r2']:.4f}")
        print("-" * 40)

        if not range_metrics or len(range_metrics) == 0:
            print("No segmented evaluation available (range_metrics is empty).")
            return

        print(f"Segments found: {len(range_metrics)}")
        print("[Evaluation by Price Range – All Features]")

        # Convert to DataFrame and validate structure
        try:
            df_metrics = pd.DataFrame(range_metrics)

            expected_cols = {"price_range", "count", "mae", "rmse", "r2"}
            if not expected_cols.issubset(df_metrics.columns):
                print(f"[ERROR] Missing expected columns in df_metrics: found {df_metrics.columns}")
                return

            # Rename and reorder
            df_metrics = df_metrics.rename(columns={
                "price_range": "Price Range",
                "count": "n",
                "mae": "MAE (€)",
                "rmse": "RMSE (€)",
                "r2": "R²"
            })
            df_metrics = df_metrics[["Price Range", "n", "MAE (€)", "RMSE (€)", "R²"]]

            print(df_metrics.to_string(index=False, float_format="{:,.2f}".format))
            print("-" * 40)
        except Exception as e:
            print(f"[ERROR] Could not format segmented metrics: {e}")






    def _compute_global_metrics(self, y_true, y_pred):
        """
        Compute overall MAE, RMSE, and R² for the entire dataset.
        """
        return {
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": self.root_mean_squared_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred)
        }

    def _compute_metrics_by_price_range(self, y_true, y_pred, bins):
        """
        Compute metrics (MAE, RMSE, R²) for each defined price segment.
        """
        df = pd.DataFrame({"true": y_true, "pred": y_pred})
        df["price_range"] = pd.cut(df["true"], bins=bins)

        results = []
        for name, group in df.groupby("price_range"):
            #if len(group) < 10:
            #    continue  # Skip small sample sizes
            results.append({
                "price_range": str(name),
                "mae": mean_absolute_error(group["true"], group["pred"]),
                "rmse": self.root_mean_squared_error(group["true"], group["pred"]),
                "r2": r2_score(group["true"], group["pred"]),
                "count": len(group)
            })
        return results

    @staticmethod
    def root_mean_squared_error(y_true, y_pred):
        """
        Compute the root of the mean squared error.
        """
        return np.sqrt(mean_squared_error(y_true, y_pred))
