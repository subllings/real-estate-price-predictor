import yaml

def load_column_mapping(filepath="configs/feature_mapping.yaml"):
    with open(filepath, "r") as f:
        return yaml.safe_load(f)

def standardize_columns(df, mapping_dict):
    """
    Renames columns in the dataframe to standardized names using a mapping dictionary.
    """
    inverse_map = {}
    for standard_col, variants in mapping_dict["columns"].items():
        for v in variants:
            inverse_map[v] = standard_col

    df = df.rename(columns=lambda col: inverse_map.get(col, col))
    return df
