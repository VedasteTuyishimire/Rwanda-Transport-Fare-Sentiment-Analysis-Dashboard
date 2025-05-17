"""
Data processing pipeline for Rwanda Transport Fare Sentiment Analysis.
"""
import os
import sys
import pandas as pd
from datetime import datetime
import logging

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collectors.twitter_collector import TwitterCollector
from models.sentiment_analyzer import SentimentAnalyzer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPipeline:
    def __init__(self):
        """Initialize the data processing pipeline."""
        self.twitter_collector = TwitterCollector()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Ensure data directories exist
        self.raw_data_dir = os.path.join('data', 'raw')
        self.processed_data_dir = os.path.join('data', 'processed')
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)
    
    def collect_data(self, days_back=7):
        """
        Collect data from all sources.
        
        Args:
            days_back (int): Number of days to look back for data collection
            
        Returns:
            pandas.DataFrame: Collected data
        """
        logger.info("Starting data collection...")
        
        # Collect Twitter data
        twitter_data = self.twitter_collector.collect_tweets(days_back=days_back)
        logger.info(f"Collected {len(twitter_data)} tweets")
        
        # Save raw data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        raw_file = os.path.join(self.raw_data_dir, f'raw_data_{timestamp}.csv')
        twitter_data.to_csv(raw_file, index=False)
        logger.info(f"Saved raw data to {raw_file}")
        
        return twitter_data
    
    def process_data(self, df):
        """
        Process the collected data.
        
        Args:
            df (pandas.DataFrame): Raw data to process
            
        Returns:
            pandas.DataFrame: Processed data with sentiment analysis
        """
        logger.info("Starting data processing...")
        
        # Perform sentiment analysis
        sentiments = self.sentiment_analyzer.analyze_batch(df['text'].tolist())
        
        # Merge sentiment results with original data
        processed_df = pd.concat([df, sentiments], axis=1)
        
        # Save processed data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        processed_file = os.path.join(
            self.processed_data_dir,
            f'processed_data_{timestamp}.csv'
        )
        processed_df.to_csv(processed_file, index=False)
        logger.info(f"Saved processed data to {processed_file}")
        
        return processed_df
    
    def run_pipeline(self, days_back=7):
        """
        Run the complete data pipeline.
        
        Args:
            days_back (int): Number of days to look back for data collection
            
        Returns:
            pandas.DataFrame: Final processed data
        """
        try:
            # Collect data
            raw_data = self.collect_data(days_back=days_back)
            
            # Process data
            processed_data = self.process_data(raw_data)
            
            logger.info("Pipeline completed successfully")
            return processed_data
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
    
    def clean_old_files(self, max_age_days=7):
        """
        Clean up old data files.
        
        Args:
            max_age_days (int): Maximum age of files to keep
        """
        logger.info(f"Cleaning files older than {max_age_days} days...")
        
        current_time = datetime.now()
        
        for directory in [self.raw_data_dir, self.processed_data_dir]:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                file_age = datetime.fromtimestamp(os.path.getctime(file_path))
                
                if (current_time - file_age).days > max_age_days:
                    os.remove(file_path)
                    logger.info(f"Removed old file: {file_path}")

if __name__ == "__main__":
    # Run the pipeline
    pipeline = DataPipeline()
    
    try:
        # Clean old files
        pipeline.clean_old_files()
        
        # Run pipeline for last 7 days
        processed_data = pipeline.run_pipeline(days_back=7)
        print(f"Processed {len(processed_data)} records successfully")
        
    except Exception as e:
        print(f"Error running pipeline: {str(e)}")
        sys.exit(1) 