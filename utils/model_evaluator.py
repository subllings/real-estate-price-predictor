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
        global_metrics, range_metrics = self.evaluate(y_true, y_pred, bins)

        print(f"\nEvaluation – {self.model_name}")
        print(f"  MAE:  {global_metrics['mae']:,.2f} €")
        print(f"  RMSE: {global_metrics['rmse']:,.2f} €")
        print(f"  R²:   {global_metrics['r2']:.4f}")
        print("-" * 40)

        if range_metrics:
            print("Price Range Evaluation:")
            for r in range_metrics:
                print(f"  {r['price_range']:<25} → "
                      f"MAE: {r['mae']:,.0f} €, "
                      f"RMSE: {r['rmse']:,.0f} €, "
                      f"R²: {r['r2']:.3f} (n={r['count']})")
            print("-" * 40)

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
            if len(group) < 10:
                continue  # Skip small sample sizes
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
