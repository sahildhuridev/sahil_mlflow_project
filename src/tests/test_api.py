import sys
import os
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.api import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "BTCUSD Forecasting API"}

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    # Ensure it returns the expected structure
    assert "recent_predictions_log" in response.json()

# We will mock data for /predict-next-hour to verify that predictions are returned
# and are scaled sensibly. This guards against the issue of un-inversed outputs.
import pytest

# Note: We avoid testing /latest-data in pure unit tests due to yfinance,
# but predict-next-hour is mocked below.


def test_predict_next_hour_scaling(monkeypatch):
    """Ensure API returns predictions in realistic price range."""
    from src.data.fetcher import DataFetcher
    from src.features.builder import FeatureBuilder
    # ensure a target scaler exists (API will attempt to load it)
    fb = FeatureBuilder("config.yaml")
    fb.scale_target(pd.Series(np.linspace(100.0, 200.0, 10)), is_training=True)

    # construct fake historical data with hourly close prices
    df = pd.DataFrame({
        'Close': np.linspace(100.0, 200.0, 200),
        'Open': np.linspace(100.0, 200.0, 200),
        'High': np.linspace(100.0, 200.0, 200),
        'Low': np.linspace(100.0, 200.0, 200),
        'Volume': np.ones(200),
    }, index=pd.date_range('2020-01-01', periods=200, freq='H'))

    monkeypatch.setattr(DataFetcher, 'update_local_data', lambda self: df)
    # patch model loading and predictions so test doesn't require real model files
    from src.models.linear_regression import LinearRegressionModel
    from src.models.arima_model import ARIMAModel
    from src.models.lstm_model import LSTMModel

    monkeypatch.setattr(LinearRegressionModel, "load_model", lambda self: None)
    monkeypatch.setattr(LinearRegressionModel, "predict", lambda self, X: pd.Series([150.0] * len(X), index=X.index))

    monkeypatch.setattr(ARIMAModel, "load_model", lambda self: None)
    monkeypatch.setattr(ARIMAModel, "predict", lambda self, steps, exog=None: [150.0] * steps)

    monkeypatch.setattr(LSTMModel, "load_model", lambda self: None)
    monkeypatch.setattr(LSTMModel, "predict", lambda self, X: [150.0])

    response = client.get("/predict-next-hour")
    assert response.status_code == 200
    data = response.json()
    assert data.get('error') is False
    # predictions should be non-negative and not tiny fractions
    for key in ['linear_regression_prediction', 'arima_prediction',
                'lstm_prediction', 'ensemble_prediction']:
        assert key in data
        assert isinstance(data[key], (int, float))
        assert 0 < data[key] < 10000

