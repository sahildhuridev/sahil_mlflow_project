mport os
import sys
import yaml
import logging
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import requests

# Configure path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.train import TrainingPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

config_path = "config.yaml"
try:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
except Exception as e:
    config = {}

API_URL = f"http://{config.get('api', {}).get('host', '0.0.0.0')}:{config.get('api', {}).get('port', 8000)}"

def hourly_prediction_job():
    """Trigger the prediction endpoint every hour to generate and log forecasts."""
    logger.info("Running hourly prediction job...")
    try:
        response = requests.get(f"{API_URL}/predict-next-hour")
        if response.status_code == 200:
            logger.info(f"Prediction successful: {response.json()}")
        else:
            logger.error(f"Prediction failed with status code {response.status_code}: {response.text}")
    except Exception as e:
        logger.error(f"Error connecting to API for hourly prediction: {e}")

def daily_retraining_job():
    """Run the training pipeline daily at midnight."""
    logger.info("Running daily retraining job...")
    try:
        pipeline = TrainingPipeline(config_path)
        pipeline.run_pipeline()
        logger.info("Daily retraining completed successfully.")
    except Exception as e:
        logger.error(f"Error during daily retraining: {e}")

if __name__ == "__main__":
    scheduler = BlockingScheduler()
    
    # Run hourly at the top of the hour (minute=0)
    scheduler.add_job(hourly_prediction_job, CronTrigger(minute=0))
    logger.info("Scheduled hourly prediction job.")
    
    # Run daily at midnight
    scheduler.add_job(daily_retraining_job, CronTrigger(hour=0, minute=0))
    logger.info("Scheduled daily retraining job.")
    
    try:
        logger.info("Starting scheduler. Press Ctrl+C to exit.")
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")
