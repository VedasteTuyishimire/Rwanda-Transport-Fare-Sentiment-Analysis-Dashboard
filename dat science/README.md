# Rwanda Transport Fare Sentiment Analysis Dashboard

## Overview
This project analyzes public sentiment regarding Rwanda's new distance-based fare system in public transport. It processes data from various sources to provide insights for policymakers through an interactive dashboard.

## Features
- Multi-source data collection (Twitter, news comments, forums)
- Sentiment analysis using state-of-the-art NLP models
- Interactive dashboard with temporal and geographical visualizations
- Trend analysis and key concerns identification
- Automated misinformation detection

## Tech Stack
- **Data Collection**: Tweepy, Selenium, BeautifulSoup4
- **Data Processing**: Pandas, NumPy
- **NLP & ML**: Transformers (BERT), spaCy, scikit-learn
- **Visualization**: Plotly, Dash
- **Deployment**: Docker, FastAPI
- **Database**: MongoDB

## Project Structure
```
├── data/               # Data storage
│   ├── raw/           # Raw collected data
│   └── processed/     # Processed datasets
├── notebooks/         # Jupyter notebooks for analysis
├── src/              # Source code
│   ├── collectors/   # Data collection scripts
│   ├── processors/   # Data processing modules
│   ├── models/      # ML models and training
│   ├── dashboard/   # Dashboard application
│   └── utils/       # Utility functions
├── tests/           # Unit tests
├── requirements.txt # Python dependencies
└── docker/         # Docker configuration
```

## Setup Instructions
1. Clone the repository
```bash
git clone https://github.com/yourusername/rwanda-transport-sentiment
cd rwanda-transport-sentiment
```

2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Set up environment variables
```bash
cp .env.example .env
# Edit .env with your API keys and configurations
```

5. Run the dashboard
```bash
python src/dashboard/app.py
```

## Data Sources
- Twitter API (tweets with relevant hashtags and keywords)
- News websites comments sections
- Public transport forums
- Government feedback channels

## Model Architecture
- BERT-based sentiment analysis model fine-tuned on Kinyarwanda text
- Topic modeling using LDA
- Named Entity Recognition for location and issue extraction
- Time series analysis for trend detection

## Dashboard Features
- Real-time sentiment tracking
- Geographic distribution of sentiments
- Top concerns and topics
- Trend analysis and forecasting
- Misinformation alert system

## Contributing
Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Authors
- TUYISHIMIRE Vedaste


## Acknowledgments
- Rwanda Transport Development Agency
- Local transport authorities
- Open-source community 