import os
import tweepy
import pandas as pd
from textblob import TextBlob
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TwitterCollector:
    def __init__(self):
        # Initialize Twitter API credentials
        self.client = tweepy.Client(
            bearer_token=os.getenv('TWITTER_BEARER_TOKEN'),
            consumer_key=os.getenv('TWITTER_API_KEY'),
            consumer_secret=os.getenv('TWITTER_API_SECRET'),
            access_token=os.getenv('TWITTER_ACCESS_TOKEN'),
            access_token_secret=os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
        )

    def get_sentiment(self, text):
        """Analyze sentiment of text using TextBlob"""
        analysis = TextBlob(text)
        # Return polarity score (-1 to 1)
        return analysis.sentiment.polarity

    def collect_tweets(self, query, max_results=100):
        """Collect tweets based on search query"""
        tweets = []
        
        try:
            # Search tweets from the past 7 days
            response = self.client.search_recent_tweets(
                query=query,
                max_results=max_results,
                tweet_fields=['created_at', 'lang', 'public_metrics']
            )

            if response.data:
                for tweet in response.data:
                    # Process each tweet
                    tweet_data = {
                        'id': tweet.id,
                        'text': tweet.text,
                        'created_at': tweet.created_at,
                        'lang': tweet.lang,
                        'retweet_count': tweet.public_metrics['retweet_count'],
                        'like_count': tweet.public_metrics['like_count'],
                        'reply_count': tweet.public_metrics['reply_count']
                    }
                    
                    # Add sentiment analysis
                    tweet_data['sentiment'] = self.get_sentiment(tweet.text)
                    tweets.append(tweet_data)

        except Exception as e:
            print(f"Error collecting tweets: {str(e)}")

        return pd.DataFrame(tweets)

    def save_tweets(self, df, filename):
        """Save tweets to CSV file"""
        os.makedirs('data', exist_ok=True)
        df.to_csv(f'data/{filename}', index=False)
        print(f"Saved {len(df)} tweets to data/{filename}")

def main():
    # Initialize collector
    collector = TwitterCollector()
    
    # Search query for Rwanda transport tweets
    query = '(Rwanda transport OR Rwanda fare OR Rwanda bus) -is:retweet lang:en'
    
    # Collect tweets
    tweets_df = collector.collect_tweets(query, max_results=100)
    
    # Save tweets
    if not tweets_df.empty:
        collector.save_tweets(tweets_df, 'rwanda_transport_tweets.csv')

if __name__ == "__main__":
    main() 