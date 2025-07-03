import os

# === Base directory (1 level up from current file) ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# === Subdirectories ===
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
CLEANED_DIR = os.path.join(DATA_DIR, "cleaned")
ML_READY_DIR = os.path.join(DATA_DIR, "ml_ready")
CLEANED_FOR_ML_DATA_DIR = os.path.join(DATA_DIR, "ml_ready")  
MODEL_OUTPUT_DIR = os.path.join(DATA_DIR, "ML")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
DB_DIR = os.path.join(BASE_DIR, "database")  
PREDICTIONS_DIR = os.path.join(BASE_DIR, "predictions")


# === Data files ===
RAW_DATA_FILE = os.path.join(RAW_DIR, "immoweb_real_estate.csv")
CLEANED_DATA_FILE = os.path.join(DATA_DIR, "immoweb_real_estate_cleaned_dataset.csv")
ML_READY_DATA_FILE = os.path.join(DATA_DIR, "immoweb_real_estate_ml_ready.csv")


ML_READY_DATA_FILE = os.path.join(ML_READY_DIR, "immoweb_real_estate_ml_ready.csv")
ML_READY_SAMPLE_XLSX = os.path.join(ML_READY_DIR, "immoweb_real_estate_ml_ready_sample10.xlsx")

# === Cleaning metadata ===
CLEANING_TYPE = "IQR + EPC cleanup + Facade trimming + Surface logic"

# === Modeling and preprocessing ===
TARGET_COLUMN = "price"
COLUMNS_TO_DROP = ["id", "url"]
N_TOP_FEATURES = 30
RANDOM_STATE = 42

# === Binning for price ranges ===
PRICE_BINS = [0, 250_000, 500_000, 750_000, 1_000_000, float("inf")]
PRICE_BIN_LABELS = ["<250k", "250k–500k", "500k–750k", "750k–1M", ">1M"]

# === Feature engineering ===
LEAK_FEATURES = ["price_per_m2"]

# === Path to the SQLite metrics database ===
METRICS_DB_PATH = os.path.join(DB_DIR, "metrics.db")

# === Dev mode ===
TEST_MODE = False

RUN_ON_AZURE = True  # Set to True if running on Azure ML
