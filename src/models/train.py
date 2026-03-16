import os
import sys
import yaml
import logging
import mlflow
import numpy as np
import pandas as pd
from datetime import datetime
from mlflow.exceptions import MlflowException

# Configure path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.fetcher import DataFetcher
from src.features.builder import FeatureBuilder
from src.models.linear_regression import LinearRegressionModel
from src.models.arima_model import ARIMAModel
from src.models.lstm_model import LSTMModel
from src.models.ensemble import EnsembleModel
from src.models.metrics import ModelMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingPipeline:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.fetcher = DataFetcher(config_path)
        self.fb = FeatureBuilder(config_path)
        
        mlflow.set_tracking_uri(self.config["paths"]["mlflow_tracking_uri"])
        mlflow.set_experiment("BTCUSD_Hourly_Forecasting")
        
        self.target_col = self.config["data"]["target_col"]
        self.sequence_length = self.config["data"]["sequence_length"]

    def run_pipeline(self):
        """Run the full training pipeline with Walk-Forward Validation."""
        logger.info("Starting training pipeline...")
        
        # 1. Fetch data
        raw_df = self.fetcher.update_local_data()
        
        if raw_df.empty:
            logger.error("No data fetched. Aborting training.")
            return
            
        # 2. Build features
        df = self.fb.create_features(raw_df)
        
        if len(df) < self.sequence_length * 2:
            logger.error("Not enough data to train.")
            return
            
        # Walk forward validation split
        train_size = int(len(df) * 0.8)
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:]
        
        logger.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

        features = [col for col in df.columns if col not in ['Target', self.target_col]]
        
        # Scale features
        train_scaled = self.fb.scale_features(train_df, features, is_training=True)
        val_scaled = self.fb.scale_features(val_df, features, is_training=False)

        # Labels (unscaled)
        y_train = train_scaled['Target']
        y_val = val_scaled['Target']

        # scale the target so that ARIMA and LSTM operate in normalized space;
        # the scaler is persisted by FeatureBuilder, and we'll inverse it later
        y_train_scaled = self.fb.scale_target(y_train, is_training=True)
        y_val_scaled = self.fb.scale_target(y_val, is_training=False)

        # MLFlow Run
        with mlflow.start_run(run_name=f"Run_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            # 3. Initialize Models
            lr_model = LinearRegressionModel()
            arima_model = ARIMAModel()
            input_size = len(features) + 1 # +1 for Close price if not in features, or just len(features)
            # Actually, features is everything except Target and target_col (Close). Wait, Close is the target sequence mostly.
            # Let's say input size is len(features)
            lstm_model = LSTMModel(input_size=len(features))
            
            # --- Train Linear Regression ---
            logger.info("Training Linear Regression...")
            # LR is trained on original target values, so no scaling/inversion
            lr_model.train(train_scaled[features], y_train)
            lr_preds = lr_model.predict(val_scaled[features])
            lr_metrics = ModelMetrics.calculate_metrics(y_val, lr_preds)
            mlflow.log_metrics({f"LR_{k}": v for k, v in lr_metrics.items()})

            # --- Train ARIMA ---
            logger.info("Training ARIMA...")
            # train on raw (unscaled) target prices — no inverse transform needed
            arima_model.train(y_train)
            arima_preds = arima_model.predict(steps=len(val_df))
            arima_metrics = ModelMetrics.calculate_metrics(y_val, arima_preds)
            mlflow.log_metrics({f"ARIMA_{k}": v for k, v in arima_metrics.items()})

            # --- Train LSTM ---
            logger.info("Training LSTM...")
            # LSTM sees scaled features and scaled target
            lstm_model.train(train_scaled[features], y_train_scaled)
            
            # For LSTM prediction, build sequences from the validation set
            lstm_preds_raw_scaled = lstm_model.predict(val_scaled[features])
            # The length of lstm_preds_raw_scaled will be len(val_scaled) - sequence_length
            y_val_lstm = y_val.iloc[self.sequence_length:]
            # inverse transform predictions back to original price scale
            lstm_preds_raw = self.fb.inverse_scale_target(lstm_preds_raw_scaled)
            
            lstm_metrics = ModelMetrics.calculate_metrics(y_val_lstm, lstm_preds_raw)
            mlflow.log_metrics({f"LSTM_{k}": v for k, v in lstm_metrics.items()})

            # --- Ensemble ---
            # To ensemble, we need aligned predictions.
            logger.info("Calculating Ensemble weights and predictions...")
            aligned_len = len(y_val_lstm)
            aligned_lr_preds = lr_preds.iloc[-aligned_len:]
            aligned_arima_preds = arima_preds[-aligned_len:]
            aligned_y_val = y_val_lstm
            
            ensemble = EnsembleModel()
            metrics_for_weights = {
                'lr': lr_metrics['mse'], 
                'arima': arima_metrics['mse'], 
                'lstm': lstm_metrics['mse']
            }
            optimal_weights = ensemble.update_weights(metrics_for_weights)
            
            mlflow.log_params(optimal_weights)
            
            ensemble_preds = []
            for i in range(aligned_len):
                pred = ensemble.predict(
                    aligned_lr_preds.iloc[i], 
                    aligned_arima_preds[i], 
                    lstm_preds_raw[i]
                )
                ensemble_preds.append(pred)
                
            ensemble_metrics = ModelMetrics.calculate_metrics(aligned_y_val, ensemble_preds)
            mlflow.log_metrics({f"Ensemble_{k}": v for k, v in ensemble_metrics.items()})
            
            roi_metrics = ModelMetrics.calculate_roi(aligned_y_val.values, ensemble_preds)
            mlflow.log_metrics({f"Ensemble_{k}": v for k, v in roi_metrics.items()})
            
            logger.info(f"Final Ensemble MSE: {ensemble_metrics['mse']:.4f}")
            logger.info(f"Final Simulated ROI: {roi_metrics['roi_percentage']:.2f}%")

            # Older copied MLflow DBs can contain stale artifact paths from another machine.
            # Training should still succeed locally even if artifact upload is unavailable.
            try:
                mlflow.log_artifact(self.config["paths"]["models_dir"])
            except (PermissionError, OSError, MlflowException) as exc:
                logger.warning("Skipping MLflow artifact upload due to local path issue: %s", exc)

if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()
