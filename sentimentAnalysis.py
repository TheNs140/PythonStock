from textblob import TextBlob
import requests
from bs4 import BeautifulSoup



def get_news_headlines(ticker):
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    
    soup = BeautifulSoup(response.text, "html.parser")
    news_table = soup.find(id="news-table")

    headlines = []
    if news_table:
        rows = news_table.find_all("tr")
        for row in rows:
            headline = row.a.text.strip()
            headlines.append(headline)

    return headlines

def analyze_sentiment(headlines):
    sentiments = []
    for headline in headlines:
        sentiment = TextBlob(headline).sentiment.polarity
        sentiments.append(sentiment)
    
    # Average sentiment for the stock
    return sum(sentiments) / len(sentiments) if sentiments else 0



# Example: Get Tesla news
news_headlines = get_news_headlines("TSLA")

# Get sentiment score for Tesla
sentiment_score = analyze_sentiment(news_headlines)
print(f"Sentiment Score for TSLA: {sentiment_score}")