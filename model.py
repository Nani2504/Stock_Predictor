import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from datetime import timedelta

# -----------------------
# Sliding Window
# -----------------------
def create_dataset(data, window=5):

    X = []
    y = []

    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])

    return np.array(X), np.array(y)

# -----------------------
# Extreme Learning Machine
# -----------------------
class ELM:

    def __init__(self, input_size, hidden_size):

        np.random.seed(42)

        self.input_weights = np.random.randn(input_size, hidden_size)
        self.bias = np.random.randn(hidden_size)

    def activation(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):

        H = self.activation(np.dot(X, self.input_weights) + self.bias)

        lambda_reg = 0.01
        I = np.eye(H.shape[1])

        self.output_weights = np.linalg.inv(H.T @ H + lambda_reg * I) @ H.T @ y

    def predict(self, X):

        H = self.activation(np.dot(X, self.input_weights) + self.bias)

        return np.dot(H, self.output_weights)


# -----------------------
# Bias Boosted ELM
# -----------------------
def bias_boosted_elm(X, y, n_models=3):

    models = []
    predictions = np.zeros(len(y))

    residual = y.copy()

    for i in range(n_models):

        elm = ELM(input_size=X.shape[1], hidden_size=20)

        elm.fit(X, residual)

        pred = elm.predict(X)

        predictions += pred

        residual = y - predictions

        models.append(elm)

    return models


# -----------------------
# Train ML Models
# -----------------------
def train_ml_models(X, y):

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    xgb = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        objective="reg:squarederror"
    )

    rf.fit(X, y)
    xgb.fit(X, y)

    return rf, xgb


# -----------------------
# BEL Bias Correction
# -----------------------
def bel_correction(predictions, actual, alpha=0.3):

    corrected = []
    emotional_weight = 0

    for i in range(len(predictions)):

        error = actual[i] - predictions[i]

        emotional_weight = emotional_weight + alpha * error

        corrected_value = predictions[i] + emotional_weight

        corrected.append(corrected_value)

    return np.array(corrected)


# -----------------------
# Train + Forecast
# -----------------------
def train_model(df):

    df["date"] = pd.to_datetime(df["date"])

    data = df["value"].values

    scaler = MinMaxScaler()

    data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()

    # Sliding window
    X, y = create_dataset(data_scaled)

    # Train models
    elm_models = bias_boosted_elm(X, y)

    rf, xgb = train_ml_models(X, y)

    history = list(data_scaled)

    predictions = []

    for _ in range(5):

        x_input = np.array(history[-5:])
        x_input = x_input.reshape(1, -1)

        # BEL / ELM prediction
        bel_pred = 0
        for model in elm_models:
            bel_pred += model.predict(x_input)[0]

        # Random Forest prediction
        rf_pred = rf.predict(x_input)[0]

        # XGBoost prediction
        xgb_pred = xgb.predict(x_input)[0]

        # Ensemble
        pred = (bel_pred + rf_pred + xgb_pred) / 3

        predictions.append(pred)

        history.append(pred)

    predictions = scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    ).flatten()

    # BEL correction
    predictions = bel_correction(predictions, data[-5:])

    last_date = df["date"].iloc[-1]

    future_dates = [last_date + timedelta(days=i) for i in range(1, 6)]

    return predictions, future_dates, data