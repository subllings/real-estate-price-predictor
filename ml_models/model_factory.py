from ml_models.rf_model import RFModel
from ml_models.lgbm_model import LGBMModel
from ml_models.lr_model import LRModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import sys
import os



class ModelFactory:
    """
    Factory class to retrieve model instances by type.
    """

    model_map = {
        "rf": RFModel,
        "lgbm": LGBMModel,
        "lr": LRModel
    }

    @staticmethod
    def create(model_type):
        if model_type == "rf":
            
            return RandomForestRegressor()
        elif model_type == "lr":
            
            return LinearRegression()
        elif model_type == "dgbm":
            


            # Redirect LightGBM stdout/stderr to null
            sys.stdout = open(os.devnull, "w")
            sys.stderr = open(os.devnull, "w")

            model = lgb.LGBMRegressor(verbose=-1)

            # Restore stdout/stderr
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__        

            return model
        

        else:
            raise ValueError(f"Unknown model type: {model_type}")
