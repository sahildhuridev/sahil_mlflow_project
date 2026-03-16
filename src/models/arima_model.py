import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import joblib
import os
import yaml
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter('ignore', ConvergenceWarning)

class ARIMAModel:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        # Using a default (5,1,0) order. For production auto_arima from pmdarima is preferred, 
        # but statsmodels ARIMA is standard and fast for baseline.
        self.order = (5, 1, 0)
        self.model_fit = None
        self.model_path = os.path.join(self.config["paths"]["models_dir"], "arima_model.pkl")

    def train(self, endog: pd.Series, exog: pd.DataFrame = None):
        # We can pass exogenous variables (rest of features) to ARIMA (making it ARIMAX)
        # But to keep it simple and robust, we might just train on the price series or use simple exog
        model = ARIMA(endog, exog=exog, order=self.order)
        self.model_fit = model.fit()
        self.save_model()

    def predict(self, steps: int = 1, exog: pd.DataFrame = None) -> list:
        if self.model_fit is None:
            self.load_model()
            
        forecast = self.model_fit.forecast(steps=steps, exog=exog)
        return forecast.tolist()

    def save_model(self):
        if self.model_fit is not None:
            # Statsmodels results can be large to pickle, wrapping standard joblib for now
            joblib.dump(self.model_fit, self.model_path)

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model_fit = joblib.load(self.model_path)
        else:
            raise FileNotFoundError(f"Model not found at {self.model_path}")
