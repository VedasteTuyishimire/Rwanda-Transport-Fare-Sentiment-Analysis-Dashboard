"""
Main script to run the Rwanda Transport Fare Sentiment Analysis system.
"""
import os
import sys
import argparse
import logging
from multiprocessing import Process
from datetime import datetime
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from processors.pipeline import DataPipeline
from dashboard.app import app as dashboard_app

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_pipeline(interval_hours=6):
    """
    Run the data pipeline at specified intervals.
    
    Args:
        interval_hours (int): Hours between pipeline runs
    """
    pipeline = DataPipeline()
    
    while True:
        try:
            logger.info("Starting pipeline run...")
            pipeline.clean_old_files()
            processed_data = pipeline.run_pipeline(days_back=7)
            logger.info(f"Pipeline completed. Processed {len(processed_data)} records")
            
            # Sleep for specified interval
            logger.info(f"Sleeping for {interval_hours} hours...")
            time.sleep(interval_hours * 3600)
            
        except Exception as e:
            logger.error(f"Error in pipeline: {str(e)}")
            time.sleep(300)  # Wait 5 minutes before retrying

def run_dashboard():
    """Run the dashboard application."""
    dashboard_app.run_server(debug=False, host='0.0.0.0', port=8050)

def main():
    parser = argparse.ArgumentParser(
        description='Rwanda Transport Fare Sentiment Analysis System'
    )
    parser.add_argument(
        '--pipeline-interval',
        type=int,
        default=6,
        help='Hours between pipeline runs (default: 6)'
    )
    parser.add_argument(
        '--dashboard-only',
        action='store_true',
        help='Run only the dashboard (no data collection)'
    )
    parser.add_argument(
        '--pipeline-only',
        action='store_true',
        help='Run only the pipeline (no dashboard)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.dashboard_only:
            logger.info("Running dashboard only...")
            run_dashboard()
        elif args.pipeline_only:
            logger.info("Running pipeline only...")
            run_pipeline(args.pipeline_interval)
        else:
            # Run both pipeline and dashboard in separate processes
            logger.info("Starting both pipeline and dashboard...")
            
            # Start pipeline process
            pipeline_process = Process(
                target=run_pipeline,
                args=(args.pipeline_interval,)
            )
            pipeline_process.start()
            
            # Start dashboard process
            dashboard_process = Process(target=run_dashboard)
            dashboard_process.start()
            
            # Wait for processes to complete
            pipeline_process.join()
            dashboard_process.join()
            
    except KeyboardInterrupt:
        logger.info("Received shutdown signal. Stopping...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"System error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 