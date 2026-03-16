import sqlite3
import os
import yaml
import logging
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo
import pytz # Backup if needed, but zoneinfo is preferred in 3.11

logger = logging.getLogger(__name__)

class PredictionLogger:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.db_path = self.config["paths"]["predictions_db"]
        self._init_db()

    @staticmethod
    def to_ist(dt):
        """Convert a datetime object to IST."""
        if dt.tzinfo is None:
            # Assume UTC if no timezone info
            dt = dt.replace(tzinfo=pytz.UTC)
        return dt.astimezone(pytz.timezone('Asia/Kolkata'))

    def _init_db(self):
        """Create the predictions table if it doesn't exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                timestamp TEXT PRIMARY KEY,
                predicted_time TEXT,
                actual_price REAL,
                linear_regression_prediction REAL,
                arima_prediction REAL,
                lstm_prediction REAL,
                ensemble_prediction REAL,
                prediction_error REAL
            )
        ''')
        conn.commit()
        conn.close()

    def log_prediction(self, timestamp: str, predicted_time: str, lr_pred: float, arima_pred: float, 
                       lstm_pred: float, ensemble_pred: float):
        """Log a new prediction. Actual price and error will be updated later."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO predictions (timestamp, predicted_time, linear_regression_prediction, 
                arima_prediction, lstm_prediction, ensemble_prediction) 
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (timestamp, predicted_time, lr_pred, arima_pred, lstm_pred, ensemble_pred))
            conn.commit()
            logger.info(f"Logged prediction for {timestamp} (target: {predicted_time})")
        except sqlite3.IntegrityError:
            # If timestamp already exists, update it
            cursor.execute('''
                UPDATE predictions 
                SET predicted_time=?, linear_regression_prediction=?, arima_prediction=?, 
                lstm_prediction=?, ensemble_prediction=?
                WHERE timestamp=?
            ''', (predicted_time, lr_pred, arima_pred, lstm_pred, ensemble_pred, timestamp))
            conn.commit()
            logger.info(f"Updated prediction for {timestamp} (target: {predicted_time})")
            
        conn.close()

    def update_actual(self, timestamp: str, actual_price: float):
        """Update actual price for a past prediction and calculate error."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT ensemble_prediction FROM predictions WHERE timestamp=?', (timestamp,))
        result = cursor.fetchone()
        
        if result:
            ensemble_pred = result[0]
            error = ensemble_pred - actual_price
            
            cursor.execute('''
                UPDATE predictions 
                SET actual_price=?, prediction_error=?
                WHERE timestamp=?
            ''', (actual_price, error, timestamp))
            conn.commit()
            logger.info(f"Updated actual price and error for {timestamp}")
        else:
            logger.warning(f"No prediction found for {timestamp} to update actual.")
            
        conn.close()

    def get_recent_logs(self, limit: int = 24):
        """Retrieve recent prediction logs."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM predictions 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]

    def get_unresolved_predictions(self):
        """Retrieve predictions that have no actual_price yet."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM predictions 
            WHERE actual_price IS NULL
            ORDER BY timestamp ASC
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
