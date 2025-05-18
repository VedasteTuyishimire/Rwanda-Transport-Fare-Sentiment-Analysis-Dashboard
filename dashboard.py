import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import base64
import io
import os

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

class SentimentDashboard:
    def __init__(self):
        self.app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.df = self.load_data()
        self.setup_layout()
        self.setup_callbacks()

    def load_data(self):
        """Load and validate tweet data"""
        file_path = 'data/rwanda_transport_tweets.csv'
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found. Please run twitter_collector.py first.")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=['text', 'created_at', 'sentiment'])
            
        df = pd.read_csv(file_path)
        if len(df) == 0:
            print("Warning: No tweets found in the CSV file.")
            return pd.DataFrame(columns=['text', 'created_at', 'sentiment'])
            
        df['created_at'] = pd.to_datetime(df['created_at'])
        return df

    def generate_wordcloud(self):
        """Generate word cloud from tweets"""
        if len(self.df) == 0:
            # Return empty base64 string or placeholder image
            return ''
            
        text = ' '.join(self.df['text'])
        stop_words = set(stopwords.words('english'))
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            stopwords=stop_words
        ).generate(text)
        
        # Convert the image to base64 string
        img = io.BytesIO()
        wordcloud.to_image().save(img, format='PNG')
        return base64.b64encode(img.getvalue()).decode()

    def setup_layout(self):
        """Set up the dashboard layout"""
        self.app.layout = dbc.Container([
            html.H1("Rwanda Transport Fare System - Public Sentiment Analysis",
                   className="text-center my-4"),
            
            # Add warning message if no data
            dbc.Alert(
                "No tweet data found. Please run twitter_collector.py first.",
                color="warning",
                id="no-data-alert",
                is_open=len(self.df) == 0,
                dismissable=True
            ),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Sentiment Overview", className="card-title"),
                            dcc.Graph(id='sentiment-pie')
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Sentiment Over Time", className="card-title"),
                            dcc.Graph(id='sentiment-time')
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Word Cloud", className="card-title"),
                            html.Img(id='wordcloud-img', 
                                   src=f'data:image/png;base64,{self.generate_wordcloud()}',
                                   style={'width': '100%'})
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Recent Tweets", className="card-title"),
                            self.create_tweet_table()
                        ])
                    ])
                ], width=12)
            ])
        ], fluid=True)

    def create_tweet_table(self):
        """Create a table of recent tweets with their sentiment"""
        if len(self.df) == 0:
            return html.P("No tweets available.")
            
        recent_tweets = self.df.sort_values('created_at', ascending=False).head(5)
        
        return html.Div([
            html.Div([
                html.P([
                    html.Strong(f"Tweet: "), tweet['text'], html.Br(),
                    html.Small(f"Sentiment: {tweet['sentiment']:.2f} | "
                             f"Created: {tweet['created_at']}")
                ], className="border p-2 mb-2")
            ]) for _, tweet in recent_tweets.iterrows()
        ])

    def setup_callbacks(self):
        @self.app.callback(
            Output('sentiment-pie', 'figure'),
            Input('sentiment-pie', 'id')
        )
        def update_sentiment_pie(_):
            if len(self.df) == 0:
                return px.pie(title='No Data Available')
                
            sentiment_counts = pd.cut(
                self.df['sentiment'],
                bins=[-1, -0.1, 0.1, 1],
                labels=['Negative', 'Neutral', 'Positive']
            ).value_counts()

            fig = px.pie(
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title='Overall Sentiment Distribution'
            )
            return fig

        @self.app.callback(
            Output('sentiment-time', 'figure'),
            Input('sentiment-time', 'id')
        )
        def update_sentiment_time(_):
            if len(self.df) == 0:
                return px.line(title='No Data Available')
                
            daily_sentiment = self.df.groupby(
                self.df['created_at'].dt.date
            )['sentiment'].mean().reset_index()

            fig = px.line(
                daily_sentiment,
                x='created_at',
                y='sentiment',
                title='Average Sentiment Over Time'
            )
            return fig

    def run(self):
        self.app.run_server(debug=True)

if __name__ == '__main__':
    dashboard = SentimentDashboard()
    dashboard.run() 