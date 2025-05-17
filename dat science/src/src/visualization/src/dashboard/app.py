"""
Dashboard application for Rwanda Transport Fare Sentiment Analysis.
"""
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
from datetime import datetime, timedelta

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

def load_latest_data():
    """Load and process the latest sentiment analysis data."""
    data_dir = os.path.join('data', 'processed')
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not files:
        return pd.DataFrame()
    
    latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(data_dir, x)))
    df = pd.read_csv(os.path.join(data_dir, latest_file))
    return df

def create_sentiment_trend():
    """Create sentiment trend over time visualization."""
    df = load_latest_data()
    if df.empty:
        return go.Figure()
    
    df['created_at'] = pd.to_datetime(df['created_at'])
    daily_sentiments = df.groupby(['created_at', 'sentiment']).size().unstack(fill_value=0)
    
    fig = go.Figure()
    for sentiment in ['positive', 'neutral', 'negative']:
        if sentiment in daily_sentiments.columns:
            fig.add_trace(go.Scatter(
                x=daily_sentiments.index,
                y=daily_sentiments[sentiment],
                name=sentiment.capitalize(),
                mode='lines+markers'
            ))
    
    fig.update_layout(
        title='Sentiment Trends Over Time',
        xaxis_title='Date',
        yaxis_title='Number of Mentions',
        template='plotly_white'
    )
    return fig

def create_sentiment_distribution():
    """Create sentiment distribution pie chart."""
    df = load_latest_data()
    if df.empty:
        return go.Figure()
    
    sentiment_counts = df['sentiment'].value_counts()
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title='Overall Sentiment Distribution',
        color_discrete_sequence=['#2ecc71', '#95a5a6', '#e74c3c']
    )
    return fig

def create_top_concerns():
    """Create visualization for top concerns/topics."""
    df = load_latest_data()
    if df.empty:
        return go.Figure()
    
    # Simple keyword-based topic extraction
    topics = {
        'Price': ['expensive', 'cost', 'price', 'fare'],
        'Distance': ['distance', 'kilometers', 'km', 'length'],
        'Fairness': ['fair', 'unfair', 'justice', 'reasonable'],
        'Implementation': ['system', 'implementation', 'process'],
        'Service': ['service', 'quality', 'bus', 'transport']
    }
    
    topic_counts = {topic: 0 for topic in topics}
    for _, row in df.iterrows():
        text = str(row['text']).lower()
        for topic, keywords in topics.items():
            if any(keyword in text for keyword in keywords):
                topic_counts[topic] += 1
    
    fig = go.Figure(data=[
        go.Bar(
            x=list(topic_counts.keys()),
            y=list(topic_counts.values())
        )
    ])
    
    fig.update_layout(
        title='Top Concerns/Topics',
        xaxis_title='Topic',
        yaxis_title='Number of Mentions',
        template='plotly_white'
    )
    return fig

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Rwanda Transport Fare Sentiment Analysis Dashboard",
                   className="text-center mb-4")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Sentiment Overview", className="card-title"),
                    dcc.Graph(id='sentiment-distribution',
                             figure=create_sentiment_distribution())
                ])
            ])
        ], width=6),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Top Concerns", className="card-title"),
                    dcc.Graph(id='top-concerns',
                             figure=create_top_concerns())
                ])
            ])
        ], width=6)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Sentiment Trends", className="card-title"),
                    dcc.Graph(id='sentiment-trend',
                             figure=create_sentiment_trend())
                ])
            ])
        ])
    ], className="mt-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Recent Feedback", className="card-title"),
                    html.Div(id='recent-feedback')
                ])
            ])
        ])
    ], className="mt-4"),
    
    dcc.Interval(
        id='interval-component',
        interval=5*60*1000,  # update every 5 minutes
        n_intervals=0
    )
], fluid=True)

# Callback to update recent feedback
@app.callback(
    Output('recent-feedback', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_recent_feedback(n):
    df = load_latest_data()
    if df.empty:
        return html.P("No data available")
    
    df['created_at'] = pd.to_datetime(df['created_at'])
    recent_df = df.sort_values('created_at', ascending=False).head(5)
    
    feedback_list = []
    for _, row in recent_df.iterrows():
        sentiment_color = {
            'positive': 'text-success',
            'neutral': 'text-secondary',
            'negative': 'text-danger'
        }.get(row['sentiment'], '')
        
        feedback_list.append(
            dbc.ListGroupItem([
                html.Small(
                    row['created_at'].strftime('%Y-%m-%d %H:%M'),
                    className="text-muted"
                ),
                html.P(row['text'], className="mb-1"),
                html.Small(
                    f"Sentiment: {row['sentiment'].capitalize()}",
                    className=sentiment_color
                )
            ])
        )
    
    return dbc.ListGroup(feedback_list)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050) 