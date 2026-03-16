import pandas as pd
import os
import yaml
import logging
from evidently.core.report import Report
from evidently.presets import DataDriftPreset
from src.features.builder import FeatureBuilder

logger = logging.getLogger(__name__)

class DriftMonitor:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.fb = FeatureBuilder(config_path)
        self.report_path = os.path.join(self.config["paths"]["models_dir"], "drift_report.html")

    def detect_drift(self, reference_data: pd.DataFrame, current_data: pd.DataFrame):
        """
        Detect data distribution shift and prediction drift.
        reference_data is usually the data the model was trained on.
        current_data is recent live data.
        """
        logger.info("Running Evidently Data Drift Detection...")
        
        # Build features for both datasets to ensure identical schema
        ref_features = self.fb.create_features(reference_data, is_training=False)
        curr_features = self.fb.create_features(current_data, is_training=False)
        
        # Drop columns not needed for drift detection
        drop_cols = ['Target', 'Close']
        ref_features.drop(columns=[col for col in drop_cols if col in ref_features.columns], inplace=True)
        curr_features.drop(columns=[col for col in drop_cols if col in curr_features.columns], inplace=True)

        data_drift_report = Report(metrics=[DataDriftPreset()])
        
        try:
            data_drift_report.run(reference_data=ref_features, current_data=curr_features)
            data_drift_report.save_html(self.report_path)
            logger.info(f"Drift report generated at {self.report_path}")
            
            # You can extract boolean flag from JSON response to alert
            result_json = data_drift_report.as_dict()
            drift_detected = result_json["metrics"][0]["result"]["dataset_drift"]
            
            if drift_detected:
                logger.warning("DATA DRIFT DETECTED! Model retraining recommended.")
            else:
                logger.info("No significant data drift detected.")
                
            return drift_detected

        except Exception as e:
            logger.error(f"Error generating drift report: {e}")
            return False
