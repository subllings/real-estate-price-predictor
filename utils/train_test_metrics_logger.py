import sys, os
project_root = os.path.abspath("../..")
sys.path.append(project_root)
import pandas as pd
import numpy as np
from datetime import datetime
import glob
import re

from utils.constants import (
    DATA_DIR,
    CLEANED_DIR,
    ML_READY_DIR,
    TEST_MODE,
)

# Create the model_train_test_logs folder if needed
MODEL_TRAIN_TEST_LOGS_DIR = os.path.join(DATA_DIR, "model_train_test_logs")
os.makedirs(MODEL_TRAIN_TEST_LOGS_DIR, exist_ok=True)

class TrainTestMetricsLogger:
    def __init__(
        self, 
        log_file=os.path.join(MODEL_TRAIN_TEST_LOGS_DIR, "metrics_train_test_log.csv"),
        cleaned_data_dir=CLEANED_DIR,
        ml_ready_dir=ML_READY_DIR
    ):
        self.log_file = log_file
        self.cleaned_data_dir = cleaned_data_dir
        self.ml_ready_dir = ml_ready_dir





    def get_latest_cleaned_file(self, dir_choice="cleaned", pattern=r"_(\d{8}_\d{4})\.csv$"):
        """
        Returns the file with the latest timestamp (YYYYMMDD_HHMM) in the filename.
        """
        folder = self.cleaned_data_dir if dir_choice == "cleaned" else self.ml_ready_dir
        files = glob.glob(os.path.join(folder, "*.csv"))
        latest_file = None
        latest_ts = None
        for f in files:
            m = re.search(pattern, os.path.basename(f))
            if m:
                ts = m.group(1)
                if latest_ts is None or ts > latest_ts:
                    latest_ts = ts
                    latest_file = f
        return latest_file

    def log(
        self,
        model_name,
        experiment_name,
        mae_train, rmse_train, r2_train,
        mae_test, rmse_test, r2_test,
        data_file=None,
        test_mode=TEST_MODE,
        dir_choice="ml_ready",
        n_features=None  
    ):
        # If data_file not provided, get latest from the right dir (default: ml_ready)
        if data_file is None:
            data_file = self.get_latest_cleaned_file(dir_choice=dir_choice)
            if data_file is None:
                raise FileNotFoundError(f"No CSV file found in {dir_choice} directory.")

        file_timestamp = None
        if os.path.isfile(data_file):
            file_timestamp = datetime.fromtimestamp(os.path.getmtime(data_file)).strftime("%Y-%m-%d %H:%M:%S")

        interp = self.interpret_fit(mae_train, mae_test, r2_train, r2_test, return_dict=True)
        r2_gap_diagnostic = interp["r2_gap_diagnostic"]

        # Calculate the ranking score based on test metrics
        ranking_score = self.run_ranking(mae_test, rmse_test, r2_test)

        data = {
            "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "model": [model_name],
            "experiment": [experiment_name],
            "mae_train": [mae_train],
            "rmse_train": [rmse_train],
            "r2_train": [r2_train],
            "mae_test": [mae_test],
            "rmse_test": [rmse_test],
            "r2_test": [r2_test],
            "r2_gap": [r2_train - r2_test],
            "r2_gap_diagnostic": [r2_gap_diagnostic],
            "n_features": [n_features], 
            "data_file": [os.path.basename(data_file)],
            "data_file_timestamp": [file_timestamp],
            "test_mode": [test_mode],
            "interpretation (r2,mae_gap)": [interp["status"]],
            "ranking_score": [ranking_score]
        }

        df = pd.DataFrame(data)
        if os.path.exists(self.log_file):
            df_existing = pd.read_csv(self.log_file)
            df = pd.concat([df_existing, df], ignore_index=True)
        df.to_csv(self.log_file, index=False)
        return df.tail(1)

    
    @staticmethod
    def interpret_fit(
        mae_train, mae_test, r2_train, r2_test,
        mae_threshold=0.2, r2_good=0.75, r2_poor=0.3, return_dict=False
    ):
        """
        Interprets the fit metrics of a regression model.

        Args:
            mae_train (float): Mean Absolute Error on the training set.
            mae_test (float): Mean Absolute Error on the test set.
            r2_train (float): R-squared on the training set.
            r2_test (float): R-squared on the test set.
            mae_threshold (float): Relative MAE difference threshold for overfitting/underfitting.
            r2_good (float): R-squared threshold above which performance is considered good.
            r2_poor (float): R-squared threshold below which performance is considered poor.
            return_dict (bool): If True, returns a dict with details. If False, returns a string label.

        Returns:
            str or dict: Interpretation status, or full dict if return_dict=True.
        """

        # 1. Type and value checks
        if not all(isinstance(arg, (int, float)) for arg in [mae_train, mae_test, r2_train, r2_test, mae_threshold, r2_good, r2_poor]):
            raise TypeError("All input metrics and thresholds must be numeric.")

        if mae_train < 0 or mae_test < 0:
            raise ValueError("MAE values cannot be negative.")

        if not (-np.inf <= r2_train <= 1) or not (-np.inf <= r2_test <= 1):
            raise ValueError("R-squared values should be between -infinity and 1.")

        # 2. Compute the relative MAE gap
        mae_gap = abs(mae_test - mae_train) / max(mae_train, 1e-6)

        # 3. Compute the R² gap
        r2_gap = r2_train - r2_test

        # 4. Define thresholds
        r2_drop_threshold = 0.15
        mae_overfit_threshold = mae_threshold
        mae_underfit_threshold = mae_threshold * 1.5



        # 5. R² gap diagnostic
        if r2_gap < 0:
            r2_gap_diagnostic = "Possible underfitting"
        elif r2_gap < 0.05:
            r2_gap_diagnostic = "Excellent generalization"
        elif r2_gap < 0.08:
            r2_gap_diagnostic = "Good generalization"
        elif r2_gap < 0.12:
            r2_gap_diagnostic = "Moderate overfitting"
        else:
            r2_gap_diagnostic = "Strong overfitting"

        # 6. Global interpretation (aligned with R² diagnostic)
        r2_min_reasonable = 0.60  # Acceptable generalization baseline

        if r2_gap > 0.15 or (mae_gap > mae_threshold and mae_test > mae_train):
            status = "overfitting"
        elif r2_gap < -0.05 and r2_test < r2_min_reasonable:
            status = "underfitting"
        elif r2_test >= r2_min_reasonable and mae_gap <= mae_threshold:
            status = "good generalization"
        else:
            status = "unstable"

        if return_dict:
            return {
                "status": status,
                "r2_gap": r2_gap,
                "r2_gap_diagnostic": r2_gap_diagnostic,
                "r2_train": r2_train,
                "r2_test": r2_test,
                "mae_train": mae_train,
                "mae_test": mae_test
            }
        else:
            return status




    def display_table(self, n_rows=None, sort_by_ranking=True):
        if not os.path.exists(self.log_file):
            print("Log file not found.")
            return None
        df = pd.read_csv(self.log_file)

        # Sort by ranking_score if present
        if "ranking_score" in df.columns and sort_by_ranking:
            df_disp = df.sort_values("ranking_score", ascending=False).head(n_rows).copy()
        else:
            df_disp = df.tail(n_rows).copy()

        # Format metrics in k€
        df_disp = self.format_metrics_k_euro(df_disp)

        # Add Rank and Best columns
        df_disp = df_disp.reset_index(drop=True)  # important to reset index and drop old index
        df_disp["Rank"] = df_disp.index + 1
        df_disp["Best"] = ""
        if not df_disp.empty:
            df_disp.at[0, "Best"] = "✔"

        # Reorder columns: Rank and Best first
        cols = df_disp.columns.tolist()
        if "Rank" in cols:
            cols = ["Rank", "Best"] + [c for c in cols if c not in ["Rank", "Best"]]
            df_disp = df_disp[cols]

        # Remove unwanted columns before styling!
        for col_to_drop in ['data_file', 'data_file_timestamp', 'test_mode', 'experiment']:
            if col_to_drop in df_disp.columns:
                df_disp = df_disp.drop(columns=[col_to_drop])

        # Columns to color
        train_cols = ['mae_train', 'rmse_train', 'r2_train']
        test_cols = ['mae_test', 'rmse_test', 'r2_test']

        # Calculate 1-based indices for CSS nth-child (no index column now)
        col_indices = {col: i+1 for i, col in enumerate(df_disp.columns)}

        # Build header styles
        custom_header_styles = []
        for col in train_cols:
            if col in col_indices:
                nth = col_indices[col]
                custom_header_styles.append({
                    'selector': f'th.col_heading.level0:nth-child({nth})',
                    'props': [('background-color', '#fff9c4')]  # pale yellow
                })
        for col in test_cols:
            if col in col_indices:
                nth = col_indices[col]
                custom_header_styles.append({
                    'selector': f'th.col_heading.level0:nth-child({nth})',
                    'props': [('background-color', '#f9c5c0')]  # light coral
                })

        # Interpretation color map
        interpretation_colors = {
            "good generalization": "background-color: #1976d2; color: white",
            "overfitting": "background-color: #1565c0; color: white",
            "underfitting": "background-color: #90caf9; color: black",
            "unstable": "background-color: #90caf9; color: black",
        }
        def highlight_interpretation(val):
            return interpretation_colors.get(val, "")

        styler = df_disp.style

        # Apply row coloring for best ranking (first, so it can be overridden)
        def color_rank(row):
            if row["Rank"] == 1:
                return ['background-color: #4caf50; color: white' for _ in row]
            elif row["Rank"] == 2:
                return ['background-color: #81c784; color: black' for _ in row]
            elif row["Rank"] == 3:
                return ['background-color: #c8e6c9; color: black' for _ in row]
            else:
                return ['' for _ in row]

        styler = styler.apply(color_rank, axis=1)

        # Then apply interpretation cell coloring (to overwrite rank colors if overlapping)
        #styler = styler.applymap(highlight_interpretation, subset=['interpretation'])
        styler = styler.map(highlight_interpretation, subset=['interpretation (r2,mae_gap)'])

        # Apply header coloring styles
        styler = styler.set_table_styles(custom_header_styles)

        # Hide the dataframe index column (0,1,2,...) in the output
        styler = styler.hide(axis='index')

        diagnostic_colors = {
            "Excellent generalization": "background-color: #81c784; color: black",
            "Good generalization": "background-color: #aed581; color: black",
            "Moderate overfitting": "background-color: #ffcc80; color: black",
            "Strong overfitting": "background-color: #ef9a9a; color: black",
            "Possible underfitting": "background-color: #90caf9; color: black"
        }

        # Appliquer la couleur à la colonne r2_gap_diagnostic
        def highlight_r2_gap_diagnostic(val):
            return diagnostic_colors.get(val, "")

        if "r2_gap_diagnostic" in df_disp.columns:
            styler = styler.map(highlight_r2_gap_diagnostic, subset=['r2_gap_diagnostic'])

        return styler


    @staticmethod
    def format_metrics_k_euro(df):
        # Format selected columns (MAE, RMSE) in thousands of euros
        cols_to_format = [
            "mae_train", "rmse_train", "mae_test", "rmse_test"
        ]
        
        for col in cols_to_format:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: TrainTestMetricsLogger.format_k_euro(x))


        # Format n_features as integer
        if "n_features" in df.columns:
            df["n_features"] = df["n_features"].apply(
                lambda x: int(float(x)) if pd.notnull(x) else np.nan
            )

        return df

    @staticmethod
    def format_k_euro(x):
        try:
            value = float(str(x).replace("k€", "").strip())
            return f"{value / 1000:.1f} k€"
        except Exception:
            return x  # return as is if conversion fails

    @staticmethod
    def format_decimal(x, precision=4):
        try:
            value = float(x)
            return f"{value:.{precision}f}"
        except Exception:
            return x  # return as is if conversion fails


    @staticmethod
    def run_ranking(mae_test, rmse_test, r2_test, weight_r2=2.0, weight_mae=1.0, weight_rmse=1.0):
        """
        Returns a numeric score for ranking runs.
        Higher is better. By default, prioritizes r2_test, penalizes high mae/rmse.
        """
        return (weight_r2 * r2_test) - (weight_mae * mae_test) - (weight_rmse * rmse_test)
