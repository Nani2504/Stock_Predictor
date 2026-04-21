from flask import Flask, render_template, request
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests
import mplfinance as mpf
import ta
from model import train_model

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

API_KEY = "1NJCXW5YQPH2SHQG"


# -----------------------------
# Market Sentiment
# -----------------------------
def analyze_market_sentiment(history, predictions):

    current_price = history[-1]
    future_price = predictions[-1]

    change_percent = ((future_price - current_price) / current_price) * 100

    if change_percent > 2:
        sentiment = "Positive"
        signal = "Bullish"
    elif change_percent < -2:
        sentiment = "Negative"
        signal = "Bearish"
    else:
        sentiment = "Neutral"
        signal = "Sideways"

    gauge = round(change_percent, 2)

    return sentiment, signal, gauge


# -----------------------------
# Candlestick Pattern
# -----------------------------
def detect_pattern(df):

    last = df.iloc[-1]

    body = abs(last["close"] - last["open"])
    candle_range = last["high"] - last["low"]

    if body < candle_range * 0.1:
        return "Doji (Market indecision)"

    if last["close"] > last["open"]:
        return "Bullish candle"

    return "Bearish candle"


# -----------------------------
# Support & Resistance
# -----------------------------
def support_resistance(df):

    support = df["low"].rolling(20).min().iloc[-1]
    resistance = df["high"].rolling(20).max().iloc[-1]

    return round(support,2), round(resistance,2)


# -----------------------------
# RSI Momentum
# -----------------------------
def calculate_rsi(df):

    rsi = ta.momentum.RSIIndicator(close=df["close"], window=21)

    return round(rsi.rsi().iloc[-1],2)


# -----------------------------
# Fetch Stock Data
# -----------------------------
def fetch_stock_data(symbol):

    url = "https://www.alphavantage.co/query"

    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": API_KEY
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "Time Series (Daily)" not in data:
        return None

    ts = data["Time Series (Daily)"]

    df = pd.DataFrame(ts).T

    df = df.rename(columns={
        "1. open": "open",
        "2. high": "high",
        "3. low": "low",
        "4. close": "close"
    })

    df["date"] = df.index

    df["open"] = df["open"].astype(float)
    df["high"] = df["high"].astype(float)
    df["low"] = df["low"].astype(float)
    df["close"] = df["close"].astype(float)

    df = df[["date","open","high","low","close"]]

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date")

    return df


# -----------------------------
# Home
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# Dataset Prediction
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["dataset"]

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    df = pd.read_csv(filepath)

    df = df.rename(columns={
        df.columns[0]:"date",
        df.columns[1]:"value"
    })

    df["date"] = pd.to_datetime(df["date"])

    predictions, future_dates, history = train_model(df)

    sentiment, signal, gauge = analyze_market_sentiment(history, predictions)

    graph_path = os.path.join(STATIC_FOLDER, "prediction.png")

    plt.figure(figsize=(10,5))

    plt.plot(history,label="History")

    plt.plot(range(len(history),len(history)+5),predictions,label="Prediction")

    plt.legend()

    plt.savefig(graph_path)

    plt.close()

    results = list(zip(future_dates,predictions))

    return render_template(
        "result.html",
        results=results,
        pred_chart="prediction.png",
        sentiment=sentiment,
        signal=signal,
        gauge=gauge
    )


# -----------------------------
# Real Time Stock Prediction
# -----------------------------
@app.route("/stock_predict", methods=["POST"])
def stock_predict():

    ticker = request.form.get("ticker")

    df = fetch_stock_data(ticker)

    if df is None:
        return "Invalid ticker OR API limit reached"

    # Prediction
    df_pred = df[["date","close"]].rename(columns={"close":"value"})

    predictions, future_dates, history = train_model(df_pred)

    sentiment, signal, gauge = analyze_market_sentiment(history, predictions)

    # Candlestick chart
    df_chart = df.copy()
    df_chart.set_index("date", inplace=True)

    candle_path = os.path.join(STATIC_FOLDER,"candlestick.png")

    mpf.plot(
        df_chart,
        type="candle",
        style="yahoo",
        mav=(5,10,20),
        volume=False,
        savefig=candle_path
    )

    # Prediction chart
    pred_path = os.path.join(STATIC_FOLDER,"prediction.png")

    plt.figure(figsize=(10,5))

    plt.plot(history,label="History")

    plt.plot(range(len(history),len(history)+5),predictions,label="Prediction")

    plt.legend()

    plt.savefig(pred_path)

    plt.close()

    # Analysis
    pattern = detect_pattern(df)

    support,resistance = support_resistance(df)

    rsi = calculate_rsi(df)

    results = list(zip(future_dates,predictions))

    return render_template(
        "result.html",
        results=results,
        candle_chart="candlestick.png",
        pred_chart="prediction.png",
        sentiment=sentiment,
        signal=signal,
        gauge=gauge,
        pattern=pattern,
        support=support,
        resistance=resistance,
        rsi=rsi
    )


if __name__ == "__main__":
    app.run(debug=True)