import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.base import clone

class ModelEvaluator:
    def __init__(self, model_name):
        # Store the model identifier for printouts
        self.model_name = model_name

    def evaluate_model(self, model, X, y, bins=None, cv=3):
        """
        Evaluate a model using cross-validation and return comprehensive metrics.
        
        Args:
            model: The sklearn-compatible model to evaluate
            X: Feature matrix
            y: Target values
            bins: Optional price bins for segmented evaluation
            cv: Number of cross-validation folds
            
        Returns:
            dict: Evaluation results including global metrics and CV scores
        """
        try:
            # Clone the model to avoid modifying the original
            model_clone = clone(model)
            
            # Fit the model on full data for predictions
            model_clone.fit(X, y)
            y_pred = model_clone.predict(X)
            
            # Get global metrics
            global_metrics = self._compute_global_metrics(y, y_pred)
            
            # Get cross-validation scores for more robust evaluation
            cv_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
            cv_rmse_scores = np.sqrt(-cv_scores)
            
            # Compile results
            evaluation_results = {
                'global_metrics': global_metrics,
                'cv_rmse_mean': cv_rmse_scores.mean(),
                'cv_rmse_std': cv_rmse_scores.std(),
                'cv_scores': cv_rmse_scores.tolist(),
                'model_name': self.model_name
            }
            
            # Add segmented metrics if bins provided
            if bins:
                range_metrics = self._compute_metrics_by_price_range(y, y_pred, bins)
                evaluation_results['range_metrics'] = range_metrics
            
            return evaluation_results
            
        except Exception as e:
            print(f"[ERROR] Model evaluation failed for {self.model_name}: {e}")
            return {
                'global_metrics': {'mae': float('inf'), 'rmse': float('inf'), 'r2': -float('inf')},
                'cv_rmse_mean': float('inf'),
                'cv_rmse_std': float('inf'),
                'cv_scores': [],
                'model_name': self.model_name,
                'error': str(e)
            }

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



    @staticmethod
    def is_model_perfect(evaluator, y_true_train, y_pred_train, y_true_test, y_pred_test) -> bool:
        global_metrics_train, _ = evaluator.evaluate(y_true_train, y_pred_train)
        global_metrics_test, _ = evaluator.evaluate(y_true_test, y_pred_test)

        r2_train = global_metrics_train["r2"]
        r2_test = global_metrics_test["r2"]
        mae_test = global_metrics_test["mae"]
        rmse_test = global_metrics_test["rmse"]

        r2_gap = abs(r2_train - r2_test)

        PERFECT_R2 = 0.90
        MAX_R2_GAP = 0.05
        MAX_MAE = 25000
        MAX_RMSE = 30000

        return (
            r2_test >= PERFECT_R2 and
            r2_gap <= MAX_R2_GAP and
            mae_test <= MAX_MAE and
            rmse_test <= MAX_RMSE
        )