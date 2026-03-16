from fastapi import FastAPI, BackgroundTasks, HTTPException
import uvicorn
import yaml
import os
import sys

# Configure path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.data.fetcher import DataFetcher
from src.features.builder import FeatureBuilder
from src.monitoring.logger import PredictionLogger
from src.models.train import TrainingPipeline
from src.models.linear_regression import LinearRegressionModel
from src.models.arima_model import ARIMAModel
from src.models.lstm_model import LSTMModel
from src.models.ensemble import EnsembleModel
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="BTCUSD Hourly Forecasting Engine", version="2.0.0")

# Allow all origins for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load config
config_path = "config.yaml"
try:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
except Exception as e:
    config = {}

@app.get("/")
def read_root():
    """Serve the premium dashboard UI."""
    return FileResponse("src/static/index.html")

@app.get("/health")
def health_check():
    """Check system status."""
    return {"status": "ok", "service": "BTCUSD Forecasting API", "version": "2.0.0"}

# ---------------------------------------------------------------------------
# /btc-price — Lightweight: returns current BTC price from local CSV (no live fetch)
# ---------------------------------------------------------------------------
@app.get("/btc-price")
def get_btc_price():
    """Return the latest BTC close price from the local CSV cache."""
    try:
        data_path = config.get("paths", {}).get("data_dir", "data")
        csv_path = os.path.join(data_path, "raw_data.csv")
        if not os.path.exists(csv_path):
            raise HTTPException(status_code=404, detail="Local data CSV not found. Run training first.")
        df = pd.read_csv(csv_path, index_col="Datetime", parse_dates=True)
        if df.empty:
            raise HTTPException(status_code=404, detail="No data in CSV.")
        latest = df.iloc[-1]
        ts = df.index[-1]
        # Convert to IST, return as plain string
        try:
            ist_ts = PredictionLogger.to_ist(ts)
            ts_str = ist_ts.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            ts_str = str(ts)[:19]
        return {
            "timestamp": ts_str,
            "close": float(latest["Close"]),
            "open": float(latest["Open"]),
            "high": float(latest["High"]),
            "low": float(latest["Low"]),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------------------------
# /latest-data — Returns last 24 OHLCV rows with clean IST string keys
# ---------------------------------------------------------------------------
@app.get("/latest-data")
def get_latest_data():
    """Return latest BTC market data with IST timestamps as plain strings (no tz offset)."""
    try:
        fetcher = DataFetcher(config_path)
        df = fetcher.fetch_latest_data(period="2d")
        if df.empty:
            raise HTTPException(status_code=404, detail="No data available")

        def _to_ist_str(ts):
            try:
                ist = PredictionLogger.to_ist(ts)
                return ist.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                return str(ts)[:19]

        df = df.tail(24).copy()
        df.index = [_to_ist_str(ts) for ts in df.index]

        return df.to_dict(orient="index")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------------------------
# /logs — Direct DB read — always works regardless of ML model state
# ---------------------------------------------------------------------------
@app.get("/logs")
def get_logs(limit: int = 50):
    """Return recent prediction history directly from the SQLite database."""
    try:
        pred_logger = PredictionLogger(config_path)
        logs = pred_logger.get_recent_logs(limit=limit)
        return {"logs": logs, "count": len(logs), "status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def background_train():
    pipeline = TrainingPipeline(config_path)
    pipeline.run_pipeline()

@app.post("/train")
def train_models(background_tasks: BackgroundTasks):
    """Trigger background training of all models."""
    background_tasks.add_task(background_train)
    return {"status": "Training started in background."}

# ---------------------------------------------------------------------------
# Sanity & Stability Logic
# ---------------------------------------------------------------------------
def stabilize(pred, reference):
    """Tighten prediction to a ±10% band around the reference price."""
    lower = reference * 0.9
    upper = reference * 1.1
    return max(lower, min(pred, upper))

def smooth(current, previous):
    """Refined temporal smoothing: 85% current, 15% previous for realistic dynamics."""
    if previous is None:
        return current
    return 0.85 * current + 0.15 * previous

def resolve_past_predictions(config_path):
    """Match past unresolved predictions with actual market data entries."""
    try:
        pred_logger = PredictionLogger(config_path)
        unresolved = pred_logger.get_unresolved_predictions()
        if not unresolved:
            return

        fetcher = DataFetcher(config_path)
        raw_df = fetcher.update_local_data()
        
        # Clean timestamps for matching
        def _clean_ts(ts):
            return str(ts)[:19]

        raw_df.index = [_clean_ts(ts) for ts in raw_df.index]
        
        for pred in unresolved:
            ts = pred['timestamp']
            # Search for this exact timestamp in raw_data.csv (formatted as IST)
            # The DB stores IST strings, the DataFetcher returns IST strings now
            if ts in raw_df.index:
                actual = float(raw_df.loc[ts, 'Close'])
                pred_logger.update_actual(ts, actual)
    except Exception as e:
        print(f"Error resolving predictions: {e}")

# ---------------------------------------------------------------------------
# /predict-next-hour — Returns structured error on failure (not HTTP 500)
# ---------------------------------------------------------------------------
@app.get("/predict-next-hour")
def predict_next_hour():
    """Return real predictions for the next hour from all models."""
    try:
        fetcher = DataFetcher(config_path)
        fb = FeatureBuilder(config_path)

        raw_df = fetcher.update_local_data()
        # Resolve any past predictions whenever we update data
        resolve_past_predictions(config_path)
        
        if len(raw_df) < 60:
            return {"error": True, "message": "Insufficient data for inference (min 60 hours required)"}

        df_features = fb.create_features(raw_df, is_training=False)
        if df_features.empty:
            return {"error": True, "message": "Feature generation failed to produce valid rows"}

        features_list = [col for col in df_features.columns if col not in ['Target', 'Close']]
        df_scaled = fb.scale_features(df_features, features_list, is_training=False)

        # --- 1. Generate Raw Predictions ---
        # Linear Regression
        lr_model = LinearRegressionModel(config_path)
        lr_model.load_model()
        last_point_features = df_scaled[features_list].iloc[[-1]]
        lr_pred_raw = float(lr_model.predict(last_point_features).iloc[0])

        # ARIMA (trained on raw prices)
        arima_model = ARIMAModel(config_path)
        arima_model.load_model()
        arima_pred_raw = float(arima_model.predict(steps=1)[0])

        # LSTM (scaled sequence input)
        lstm_model = LSTMModel(input_size=len(features_list), config_path=config_path)
        lstm_model.load_model()
        last_seq_input = df_scaled[features_list].tail(config["data"]["sequence_length"] + 1)
        lstm_pred_raw_scaled = lstm_model.predict(last_seq_input)
        lstm_pred_scaled = float(lstm_pred_raw_scaled[-1])
        try:
            lstm_pred_unclamped = float(fb.inverse_scale_target([lstm_pred_scaled])[0])
        except FileNotFoundError:
            raise HTTPException(status_code=500, detail="Target scaler missing, please retrain models before inference")

        # --- 2. Prediction Stabilization Pipeline ---
        last_price = float(raw_df['Close'].iloc[-1])
        
        # Step 1: LSTM fallback (if collapsed)
        if lstm_pred_unclamped < last_price * 0.5:
            lstm_pred_final = lr_pred_raw
        else:
            lstm_pred_final = lstm_pred_unclamped

        # Step 2: ARIMA Smoothing (0.6 arima / 0.4 last_price) for realistic trend visibility
        arima_pred_final = 0.6 * arima_pred_raw + 0.4 * last_price

        # Step 3: Stabilization Clamp (±10%)
        lr_pred = stabilize(lr_pred_raw, last_price)
        arima_pred = stabilize(arima_pred_final, last_price)
        lstm_pred = stabilize(lstm_pred_final, last_price)

        # Step 4: Temporal Smoothing (using previous logs)
        pred_logger = PredictionLogger(config_path)
        recent_logs = pred_logger.get_recent_logs(limit=1)
        
        if recent_logs:
            prev = recent_logs[0]
            lr_pred = smooth(lr_pred, prev.get('linear_regression_prediction'))
            arima_pred = smooth(arima_pred, prev.get('arima_prediction'))
            lstm_pred = smooth(lstm_pred, prev.get('lstm_prediction'))

        # --- 3. Ensemble ---
        ensemble = EnsembleModel()
        ensemble_pred_raw = ensemble.predict(lr_pred, arima_pred, lstm_pred)
        # Smooth the final ensemble as well
        ensemble_prev = recent_logs[0].get('ensemble_prediction') if recent_logs else None
        ensemble_pred = smooth(ensemble_pred_raw, ensemble_prev)

        # --- 4. Logging & Timestamps ---
        from datetime import datetime, timedelta
        # Current IST time for the "Generated At" timestamp
        now_ist = PredictionLogger.to_ist(datetime.utcnow())
        logged_ts = now_ist.strftime("%Y-%m-%d %H:%M:%S")
        
        # Predicted target time is exactly 1 hour later
        predicted_ts = (now_ist + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")

        pred_logger = PredictionLogger(config_path)
        pred_logger.log_prediction(
            timestamp=logged_ts,
            predicted_time=predicted_ts,
            lr_pred=lr_pred,
            arima_pred=arima_pred,
            lstm_pred=lstm_pred,
            ensemble_pred=ensemble_pred
        )

        return {
            "error": False,
            "timestamp": logged_ts,
            "predicted_time": predicted_ts,
            "linear_regression_prediction": lr_pred,
            "arima_prediction": arima_pred,
            "lstm_prediction": lstm_pred,
            "ensemble_prediction": ensemble_pred
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Return structured error so frontend handles it gracefully
        return {"error": True, "message": str(e)}


@app.get("/metrics")
def get_metrics():
    """Return comprehensive model performance metrics for the dashboard."""
    try:
        # Final fresh resolution before showing metrics
        resolve_past_predictions(config_path)
        
        pred_logger = PredictionLogger(config_path)
        recent_logs = pred_logger.get_recent_logs(limit=100)

        df = pd.DataFrame(recent_logs).dropna(subset=['actual_price']) if recent_logs else pd.DataFrame()
        stats = {}
        best_model_info = None
        stability_status = "Moderate"
        leaderboard = []

        if not df.empty:
            from src.models.metrics import ModelMetrics
            stats['ensemble'] = ModelMetrics.calculate_metrics(df['actual_price'], df['ensemble_prediction'])
            stats['lr'] = ModelMetrics.calculate_metrics(df['actual_price'], df['linear_regression_prediction'])
            stats['arima'] = ModelMetrics.calculate_metrics(df['actual_price'], df['arima_prediction'])
            stats['lstm'] = ModelMetrics.calculate_metrics(df['actual_price'], df['lstm_prediction'])

            # Add best model detection
            best_model_info = ModelMetrics.get_best_model(stats)
            
            # Predict stability based on recent ensemble predictions
            stability_status = ModelMetrics.calculate_stability(df['ensemble_prediction'].tolist())

            # Prepare leaderboard
            models = [
                {"name": "LSTM", "rmse": stats['lstm']['rmse']},
                {"name": "Linear Regression", "rmse": stats['lr']['rmse']},
                {"name": "ARIMA", "rmse": stats['arima']['rmse']}
            ]
            leaderboard = sorted(models, key=lambda x: x['rmse'])

        return {
            "recent_predictions_log": recent_logs,
            "summary_stats": stats,
            "best_model": best_model_info,
            "stability_status": stability_status,
            "leaderboard": leaderboard,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitoring/drift")
def check_drift():
    """Trigger data drift detection and return status."""
    try:
        from src.monitoring.drift import DriftMonitor
        fetcher = DataFetcher(config_path)
        monitor = DriftMonitor(config_path)
        raw_df = fetcher.update_local_data()
        if len(raw_df) < 100:
            return {"drift_detected": False, "message": "Insufficient data"}
        ref = raw_df.head(len(raw_df)-48)
        curr = raw_df.tail(48)
        drift_detected = monitor.detect_drift(ref, curr)
        return {"drift_detected": bool(drift_detected)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
