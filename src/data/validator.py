import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class DataValidator:
    def __init__(self, config_path="config.yaml"):
        # We can add dynamic thresholds here from config if needed
        pass

    def validate(self, df: pd.DataFrame) -> bool:
        """
        Run full validation suite on the dataframe.
        Returns True if data is acceptable, raises ValueError or returns False if critical issues found.
        """
        logger.info("Starting data validation...")
        
        try:
            self.check_missing_values(df)
            self.check_outliers(df)
            self.check_timestamp_continuity(df)
            self.check_distribution_sanity(df)
            logger.info("Data validation passed successfully.")
            return True
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            return False

    def check_missing_values(self, df: pd.DataFrame):
        """Detect missing values."""
        missing = df.isnull().sum()
        if missing.any():
            logger.warning(f"Missing values detected: \n{missing[missing > 0]}")
            # We don't necessarily abort here if ffill/dropna handles it later, 
            # but we should log it. If too many are missing, we should abort.
            total_missing = missing.sum()
            if total_missing > len(df) * 0.1: # 10% threshold
                raise ValueError(f"Too many missing values: {total_missing} ({total_missing/len(df):.2%})")

    def check_outliers(self, df: pd.DataFrame):
        """Detect price and volume outliers using Z-score or modified Z-score."""
        for col in ['Close', 'Volume']:
            if col not in df.columns: continue
            
            # Simple Z-score outlier detection (threshold=5 for crypto)
            z_scores = np.abs((df[col] - df[col].mean()) / (df[col].std() + 1e-9))
            outliers = (z_scores > 5).sum()
            
            if outliers > 0:
                logger.warning(f"Detected {outliers} outliers in {col} (Z-score > 5)")
                if outliers > len(df) * 0.05: # 5% threshold
                     raise ValueError(f"High number of outliers in {col}: {outliers}")

    def check_timestamp_continuity(self, df: pd.DataFrame):
        """Verify timestamps are continuous (hourly)."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Index is not DatetimeIndex")
            
        # Sort just in case
        df = df.sort_index()
        
        # Calculate time difference between rows
        time_diffs = df.index.to_series().diff().dropna()
        expected_diff = pd.Timedelta(hours=1)
        
        gaps = (time_diffs != expected_diff).sum()
        if gaps > 0:
            logger.warning(f"Detected {gaps} gaps in timestamp continuity.")
            # For crypto, exchanges sometimes go down. A small number of gaps is expected.
            if gaps > len(df) * 0.05: # 5% threshold
                raise ValueError(f"Too many gaps in data continuity: {gaps}")

    def check_distribution_sanity(self, df: pd.DataFrame):
        """Basic sanity checks on value ranges."""
        # BTC price shouldn't be negative or zero
        if (df['Close'] <= 0).any():
            raise ValueError("Found non-positive values in Close price.")
            
        # High should be >= Low
        if (df['High'] < df['Low']).any():
             raise ValueError("Found records where High is less than Low.")
             
        # High should be >= Close and Open
        if (df['High'] < df['Close']).any() or (df['High'] < df['Open']).any():
             raise ValueError("Found records where High is less than Close or Open.")
