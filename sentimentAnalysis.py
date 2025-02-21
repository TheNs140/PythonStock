import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import datetime

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def get_stock_data(ticker, start, end):
    stock = yf.Ticker(ticker)
    df = stock.history(start=start, end=end)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.index = df.index.date  # Convert to just date format
    return df

def get_news_headlines(ticker, date):
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    soup = BeautifulSoup(response.text, "html.parser")
    news_table = soup.find(id="news-table")

    headlines = []
    if news_table:
        rows = news_table.find_all("tr")
        for row in rows:
            timestamp = row.td.text.strip()
            headline = row.a.text.strip()

            if ":" in timestamp:  # Ensuring it's a time, not a date
                headlines.append(headline)

    return headlines

def analyze_sentiment(headlines):
    sentiment_scores = [sia.polarity_scores(h)['compound'] for h in headlines]
    return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

def get_daily_sentiment(ticker, start, end):
    sentiment_data = {}

    start_date = datetime.datetime.strptime(start, "%Y-%m-%d").date()
    end_date = datetime.datetime.strptime(end, "%Y-%m-%d").date()
    
    current_date = start_date
    while current_date <= end_date:
        headlines = get_news_headlines(ticker, current_date)
        sentiment_score = analyze_sentiment(headlines)
        sentiment_data[current_date] = sentiment_score
        current_date += datetime.timedelta(days=1)
    
    return pd.DataFrame(list(sentiment_data.items()), columns=['Date', 'Sentiment']).set_index('Date')

def prepare_data(ticker, start, end):
    stock_data = get_stock_data(ticker, start, end)
    sentiment_data = get_daily_sentiment(ticker, start, end)

    # Merge sentiment with stock data
    stock_data = stock_data.merge(sentiment_data, left_index=True, right_index=True, how='left')
    stock_data.fillna(0, inplace=True)  # Fill missing sentiment values with 0
    return stock_data

def train_model(stock_data):
    features = stock_data[['Open', 'High', 'Low', 'Volume', 'Sentiment']]
    labels = stock_data['Close']
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    model = GaussianMixture(n_components=2, random_state=0)
    model.fit(X_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Mean Absolute Error: {mae}")

    return model

def predict_next_day_price(model, last_data):
    last_data = last_data[['Open', 'High', 'Low', 'Volume', 'Sentiment']].values.reshape(1, -1)
    return model.predict(last_data)[0]

if __name__ == "__main__":
    ticker = "AAPL"
    start_date = "2023-01-01"
    end_date = "2024-01-01"

    stock_data = prepare_data(ticker, start_date, end_date)
    model = train_model(stock_data)

    next_day_price = predict_next_day_price(model, stock_data.iloc[-1])
    print(f"Predicted Next Day Price for {ticker}: {next_day_price}")
