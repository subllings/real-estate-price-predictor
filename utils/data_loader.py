import os,sys
import glob
# Add the project root to the Python path
project_root = os.path.abspath("../..")
sys.path.append(project_root)

import pandas as pd
from utils.constants import LEAK_FEATURES, CLEANED_FOR_ML_DATA_DIR


class DataLoader:
    def __init__(self, path):
        self.path = path


    @staticmethod
    def get_latest_cleaned_file_path():
        """
        Return the name of the most recent cleaned CSV file in the given directory.
        """
        # Get list of matching CSV files
        csv_files = glob.glob(os.path.join(CLEANED_FOR_ML_DATA_DIR, "immoweb_real_estate_cleaned_for_ml_*.csv"))
        if not csv_files:
            print("No cleaned CSV files found.")
            return None
        
        # Sort by modification time and return the most recent one
        latest_file = max(csv_files, key=os.path.getmtime)
        return latest_file


    def load_data(self):
        """
        Loads data from a CSV file specified by self.path.

        Returns:
          pandas.DataFrame: The loaded DataFrame.
        """
        df = pd.read_csv(self.path)
        return df
    
    
    def clean_booleans(self, df, bool_cols):
        """
        Converts string boolean columns to integers (1/0).

        Parameters:
          df (pd.DataFrame): The input DataFrame.
          bool_cols (list): List of boolean column names to clean.

        Returns:
          pd.DataFrame: DataFrame with cleaned boolean columns.
        """
        for col in bool_cols:
            df[col] = df[col].astype(str).str.lower().map({'true': 1, 'false': 0})
            df[col] = df[col].fillna(0).astype(int)
        return df

    def drop_columns(self, df, columns_to_drop):
        """
        Drops the specified columns from the DataFrame.

        Parameters:
          df (pd.DataFrame): The input DataFrame.
          columns_to_drop (list): List of column names to drop.

        Returns:
          pd.DataFrame: DataFrame without the dropped columns.
        """
        df = df.drop(columns=columns_to_drop, errors="ignore")
        return df

    def drop_na_targets(self, df, target_col="price"):
        """
        Drops rows with missing values in the target column.

        Parameters:
          df (pd.DataFrame): The input DataFrame.
          target_col (str): Name of the target column.

        Returns:
          pd.DataFrame: DataFrame with rows containing NaN in target dropped.
        """
        return df.dropna(subset=[target_col])

    def add_price_per_m2(self, df):
        """
        Adds a feature 'price_per_m2' based on price / habitableSurface.
        
        Parameters:
          df (pd.DataFrame): The input DataFrame.

        Returns:
          pd.DataFrame: Updated DataFrame with new feature.
        """
        df["price_per_m2"] = df["price"] / df["habitableSurface"]
        return df

    def add_building_age(self, df):
        """
        Adds a feature 'building_age' based on 2025 - construction year.

        Parameters:
          df (pd.DataFrame): The input DataFrame.

        Returns:
          pd.DataFrame: Updated DataFrame with new feature.
        """
        df["building_age"] = 2025 - df["buildingConstructionYear"]
        return df

    def remove_leak_features(self, df):
        """
        Removes known features that leak information from the target.

        Parameters:
          df (pd.DataFrame): The input DataFrame.

        Returns:
          pd.DataFrame: Cleaned DataFrame without leak features.
        """
        return df.drop(columns=[col for col in LEAK_FEATURES if col in df.columns], errors="ignore")

    def split_X_y(self, df, target_column="price"):
        """
        Splits a DataFrame into features (X) and target (y) based on the specified target column.

        Parameters:
          df (pd.DataFrame): The input DataFrame containing features and target.
          target_column (str, optional): The name of the target column to separate. Defaults to "price".

        Returns:
          Tuple[pd.DataFrame, pd.Series]: A tuple containing the features DataFrame (X) and the target Series (y).

        Notes:
          - Feature column names are sanitized by replacing non-alphanumeric characters with underscores.
          - Duplicate columns in X are removed.
        """
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X.columns = X.columns.str.replace('[^a-zA-Z0-9_]', '_', regex=True)
        X = X.loc[:, ~X.columns.duplicated()]
        return X, y
