# Prerequisites

Make sure **Python 3.12.10** is installed.

- [Download Python 3.12.10](https://www.python.org/downloads/release/python-31210/)
- Add Python to your system `PATH`

> ⚠️ This project is validated under Python 3.12.10. Other versions may result in compatibility issues.

# Real Estate Data Analysis & Price Predictor

A modular and extensible project for predicting real estate prices across multiple cities and model types.  
Compatible with both **local environments** and **Databricks** (MLflow, DBFS, Delta).



## Environment Setup

Use the virtual environment to manage dependencies cleanly:

```bash
chmod +x setup-env.sh
./setup-env.sh
```


##  Project Structure

```text

real-estate-price-predictor/
├── configs/                         # YAML configuration files (e.g. column mapping)
│   └── feature_mapping.yaml         # Mapping of column variants to standard names
│
├── data/                            # Input CSV datasets
│   ├── cleaned/                     # Cleaned datasets ready for modeling
│   ├── ml_ready/                    # ML-ready final datasets
│   ├── raw/                         # Original raw exports
│   └── outputs/                     # Any intermediate or test outputs
│
├── database/                        # SQLite database storage for evaluation and cleaning logs
│   └── metrics.db                   # Centralized database for model metrics and data cleaning versions
│
├── dbfs_models/                     # Output directory for models if using Databricks
│
├── local_models/                    # Trained models stored locally
│   ├── rf/                          # Random Forest models by dataset
│   ├── lgbm/                        # LightGBM models by dataset
│   └── lr/                          # Linear Regression models by dataset
│
├── ml_models/                       # Core machine learning model definitions
│   ├── __init__.py                  # Makes this a Python package
│   ├── base_model.py                # Abstract base class for model interfaces
│   ├── rf_model.py                  # Random Forest implementation
│   ├── lgbm_model.py                # LightGBM implementation (Distributed Gradient Boosting Machine)
│   ├── lr_model.py                  # Linear Regression implementation
│   └── model_factory.py             # Factory to retrieve the correct model class
│
├── notebooks/                       # Jupyter notebooks for exploration and training
│   ├── exploration/                 # Notebooks for EDA per source
│   └── pipeline/                    # Modular notebooks (cleaning, training, tuning, export,etc.)
│       ├── 00_setup_env.ipynb           # Setup virtual environment and dependencies
│       ├── 01_exploration.ipynb         # Data exploration and inspection
│       ├── 02_preprocessing.ipynb       # Data cleaning and feature engineering
│       ├── 03_train_model.ipynb         # Training individual models
│       ├── 04_evaluate_model.ipynb      # Evaluation metrics and visualizations
│       ├── 05_register_model.ipynb      # Optional model registry logic
│       └── 06_batch_train_all.ipynb     # Loop training over all datasets
│
├── scripts/                         # Executable Python scripts
│   ├── train_all_datasets.py        # Main script to train all models for all datasets
│   ├── train_all_datasets.sh        # Bash script to launch training from terminal
│   ├── train_and_register.py        # Alternate script to train and register models
│   └── train_and_register.sh        # Bash wrapper for above
│
├── tests/                           # Unit tests
│   ├── __init__.py                  # Init file for test package
│   └── test_model_training.py       # Basic test for training pipeline
│
├── utils/                           # Utility scripts and helpers
│   ├── column_mapper.py             # Logic to standardize columns across datasets
│   ├── constants.py                 # Global constants (e.g., target column)
│   ├── logger.py                    # Logging utilities
│   ├── paths.py                     # Helper functions for path management
│   ├── model_evaluator.py           # Centralized logic to log model evaluations (MAE, RMSE, R<sup>2</sup>) to SQLite
│   ├── data_cleaner.py              # Cleans data and logs decisions (outliers, filters, price range, etc.)
│   └── preprocessing.py             # Custom preprocessing functions
│
├── .gitignore                       # Git ignored files list
├── README.md                        # Project overview and documentation
├── requirements.txt                 # Python package dependencies
└── setup-env.sh                     # Script to initialize virtual environment


real-estate-price-predictor/
├── configs/                         # YAML configuration files (e.g. column mapping)
│   └── feature_mapping.yaml         # Mapping of column variants to standard names
│
├── data/                            # Input CSV datasets
│   └── immovlan_real_estate.csv     # Sample real estate dataset
│
├── dbfs_models/                     # Output directory for models if using Databricks
│
├── local_models/                    # Trained models stored locally
│   ├── rf/                          # Random Forest models by dataset
│   ├── lgbm/                        # LightGBM models by dataset
│   └── lr/                          # Linear Regression models by dataset
│
├── ml_models/                       # Core machine learning model definitions
│   ├── __init__.py                  # Makes this a Python package
│   ├── base_model.py                # Abstract base class for model interfaces
│   ├── rf_model.py                  # Random Forest implementation
│   ├── lgbm_model.py                # LightGBM implementation (Distributed Gradient Boosting Machine)
│   ├── lr_model.py                  # Linear Regression implementation
│   └── model_factory.py             # Factory to retrieve the correct model class
│
├── notebooks/                       # Jupyter notebooks for exploration and training
│   ├── 00_setup_env.ipynb           # Setup virtual environment and dependencies
│   ├── 01_exploration.ipynb         # Data exploration and inspection
│   ├── 02_preprocessing.ipynb       # Data cleaning and feature engineering
│   ├── 03_train_model.ipynb         # Training individual models
│   ├── 04_evaluate_model.ipynb      # Evaluation metrics and visualizations
│   ├── 05_register_model.ipynb      # Optional model registry logic
│   └── 06_batch_train_all.ipynb     # Loop training over all datasets
│
├── scripts/                         # Executable Python scripts
│   ├── train_all_datasets.py        # Main script to train all models for all datasets
│   ├── train_all_datasets.sh        # Bash script to launch training from terminal
│   ├── train_and_register.py        # Alternate script to train and register models
│   └── train_and_register.sh        # Bash wrapper for above
│
├── tests/                           # Unit tests
│   ├── __init__.py                  # Init file for test package
│   └── test_model_training.py       # Basic test for training pipeline
│
├── utils/                           # Utility scripts and helpers
│   ├── column_mapper.py             # Logic to standardize columns across datasets
│   ├── constants.py                 # Global constants (e.g., target column)
│   ├── logger.py                    # Logging utilities
│   ├── paths.py                     # Helper functions for path management
│   └── preprocessing.py             # Custom preprocessing functions
│
├── .gitignore                       # Git ignored files list
├── README.md                        # Project overview and documentation
├── requirements.txt                 # Python package dependencies
└── setup-env.sh                     # Script to initialize virtual environment
```


#  Real Estate Price Prediction API

## What does the API do?

- Loads two trained **CatBoost** models (`.pkl`) at startup:
  - `catboost_optuna_all_*.pkl`: trained with **all engineered features**
  - `catboost_optuna_top30_*.pkl`: trained with **top 30 features only**
- Provides two **POST endpoints** to make predictions based on input data
- Returns the predicted price as a JSON response

## Models Used

Both models were trained using **CatBoost** with **Optuna hyperparameter tuning** and saved using `joblib`.

These models are located in:

```
app/backend/models/pkl/
├── catboost_optuna_all_{date_time}.pkl
└── catboost_optuna_top30_{date_time}.pkl
```

## Run the API

From the root of the project, start the FastAPI server using:

```bash
./run_api.sh
```

## API Endpoints

### Swagger UI

You can explore and test the API interactively via Swagger:  
`http://localhost:8000/docs`

## Test the API with Postman

### What we can do with Postman:
- Send **POST**, **GET**, etc. requests to your FastAPI endpoints
- Easily test `/predict_all` or `/predict_top30`
- View the **JSON response** returned by the real estate model
- Manage **headers**, **authentication**, and **JSON bodies** effortlessly
- Save request collections for future testing

### Installation
Download here: [https://www.postman.com/downloads/](https://www.postman.com/downloads/)


### Example test in Postman (for `/predict_top30`)

1. Open **Postman**  
2. Select request type: `POST`  
3. URL: `http://localhost:8000/predict_top30`  
4. Go to the `Headers` tab and add:
   - **Key**: `Content-Type`  
   - **Value**: `application/json`  
5. Go to the `Body` tab:
   - Choose `raw` and select `JSON` as the format
   - Paste the following example payload (simplified):


```json
{
  "habitableSurface": 120,
  "bathroomCount": 2,
  "postCode": 1000,
  "toiletCount": 2,
  "buildingConstructionYear": 2005,
  "locality_Knokke_Heist": 0,
  "building_age": 19,
  "surface_per_room": 30,
  "facedeCount": 2,
  "kitchenType_HYPER_EQUIPPED": 1,
  "buildingCondition_AS_NEW": 0,
  "province_West_Flanders": 0,
  "subtype_VILLA": 0,
  "subtype_HOUSE": 1,
  "province_Hainaut": 0,
  "room_count": 4,
  "bedroomCount": 3,
  "buildingCondition_TO_RENOVATE": 0,
  "epcScore_B": 1,
  "hasTerrace": 1,
  "subtype_PENTHOUSE": 0,
  "epcScore_C": 0,
  "buildingCondition_GOOD": 1,
  "heatingType_nan": 0,
  "hasLivingRoom": 1,
  "locality_Ixelles": 0,
  "kitchenType_INSTALLED": 0,
  "epcScore_A": 0,
  "epcScore_F": 0,
  "locality_Gent": 0
}
```
![picture 3](images/5d7f4edacdfa4c64ebf0a6d7428dc61a77620ade0f33538b594b9a652fd2b0ae.png)  
![alt text](image.png) 


### Example test in Postman (for `/predict_all`)

```json
{
  "bedroomCount": 3,
  "bathroomCount": 2,
  "postCode": 1050,
  "habitableSurface": 100,
  "buildingConstructionYear": 2000,
  "facedeCount": 2,
  "toiletCount": 2,
  "room_count": 5,
  "surface_per_room": 20,
  "building_age": 24,
  "type_APARTMENT": 1,
  "type_HOUSE": 0,
  "subtype_APARTMENT": 1,
  "subtype_APARTMENT_BLOCK": 0,
  "subtype_DUPLEX": 0,
  "subtype_GROUND_FLOOR": 0,
  "subtype_HOUSE": 0,
  "subtype_MIXED_USE_BUILDING": 0,
  "subtype_PENTHOUSE": 0,
  "subtype_TOWN_HOUSE": 0,
  "subtype_VILLA": 0,
  "province_Antwerp": 0,
  "province_Brussels": 1,
  "province_East_Flanders": 0,
  "province_Flemish_Brabant": 0,
  "province_Hainaut": 0,
  "province_Limburg": 0,
  "province_Liège": 0,
  "province_Luxembourg": 0,
  "province_Namur": 0,
  "province_Walloon_Brabant": 0,
  "province_West_Flanders": 0,
  "locality_Anderlecht": 0,
  "locality_Antwerpen": 0,
  "locality_Bruxelles": 1,
  "locality_Gent": 0,
  "locality_Ixelles": 0,
  "locality_Knokke_Heist": 0,
  "locality_Liège": 0,
  "locality_Uccle": 0,
  "buildingCondition_AS_NEW": 0,
  "buildingCondition_GOOD": 1,
  "buildingCondition_JUST_RENOVATED": 0,
  "buildingCondition_TO_BE_DONE_UP": 0,
  "buildingCondition_TO_RENOVATE": 0,
  "buildingCondition_nan": 0,
  "floodZoneType_NON_FLOOD_ZONE": 1,
  "floodZoneType_POSSIBLE_FLOOD_ZONE": 0,
  "floodZoneType_RECOGNIZED_FLOOD_ZONE": 0,
  "floodZoneType_nan": 0,
  "heatingType_ELECTRIC": 0,
  "heatingType_FUELOIL": 0,
  "heatingType_GAS": 1,
  "heatingType_PELLET": 0,
  "heatingType_nan": 0,
  "kitchenType_HYPER_EQUIPPED": 1,
  "kitchenType_INSTALLED": 0,
  "kitchenType_NOT_INSTALLED": 0,
  "kitchenType_SEMI_EQUIPPED": 0,
  "kitchenType_USA_HYPER_EQUIPPED": 0,
  "kitchenType_USA_INSTALLED": 0,
  "kitchenType_nan": 0,
  "epcScore_A": 0,
  "epcScore_A_": 0,
  "epcScore_B": 1,
  "epcScore_C": 0,
  "epcScore_D": 0,
  "epcScore_E": 0,
  "epcScore_F": 0,
  "epcScore_G": 0,
  "hasLivingRoom": 1,
  "hasTerrace": 1
}
```