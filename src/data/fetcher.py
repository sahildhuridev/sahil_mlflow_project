import yfinance as yf
import pandas as pd
import yaml
import os
import logging

from src.data.validator import DataValidator

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class DataFetcher:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.validator = DataValidator(config_path)
        self.symbol = self.config["data"]["symbol"]
        self.interval = self.config["data"]["interval"]
        self.history_period = self.config["data"]["history_period"]
        self.data_dir = self.config["paths"]["data_dir"]
        
        os.makedirs(self.data_dir, exist_ok=True)
        self.file_path = os.path.join(self.data_dir, "raw_data.csv")
        
    def fetch_historical_data(self) -> pd.DataFrame:
        """Fetch long historical data for initial training."""
        logger.info(f"Fetching historical data for {self.symbol}...")
        ticker = yf.Ticker(self.symbol)
        df = ticker.history(period=self.history_period, interval=self.interval)
        
        if df.empty:
            logger.warning("Fetched data is empty. Check internet or yfinance status.")
            return pd.DataFrame()
            
        df = self._clean_data(df)
        self.save_data(df)
        logger.info(f"Historical data fetched successfully. Shape: {df.shape}")
        return df

    def fetch_latest_data(self, period="5d") -> pd.DataFrame:
        """Fetch latest data to append to existing dataset."""
        logger.info(f"Fetching latest data for {self.symbol}...")
        ticker = yf.Ticker(self.symbol)
        df = ticker.history(period=period, interval=self.interval)
        
        if df.empty:
            return pd.DataFrame()
            
        return self._clean_data(df)

    def update_local_data(self) -> pd.DataFrame:
        """Update existing local data with new fetched data and return the full dataset."""
        if not os.path.exists(self.file_path):
            logger.info("Local data not found. Fetching historical data.")
            return self.fetch_historical_data()
            
        # Load local
        logger.info("Loading local historical data.")
        local_df = pd.read_csv(self.file_path, index_col='Datetime', parse_dates=True)
        
        # Fetch new
        new_df = self.fetch_latest_data()
        if new_df.empty:
            return local_df
            
        # Combine
        combined_df = pd.concat([local_df, new_df])
        
        # Remove duplicates based on index
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        combined_df.sort_index(inplace=True)
        
        self.save_data(combined_df)
        logger.info(f"Local data updated. New Shape: {combined_df.shape}")
        return combined_df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean dataset by removing timezone info from index and unnecessary columns."""
        if df.index.tz is not None:
            df.index = df.index.tz_convert('UTC').tz_localize(None)
            
        cols_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df.loc[:, cols_to_keep].copy()
        # Avoid chained-assignment warnings by working on a fresh frame.
        df = df.dropna(how='all')
        df = df.ffill()
        
        # Validate data
        self.validator.validate(df)
        
        return df
        
    def save_data(self, df: pd.DataFrame):
        """Save data to CSV."""
        df.to_csv(self.file_path)

if __name__ == "__main__":
    fetcher = DataFetcher()
    fetcher.update_local_data()
