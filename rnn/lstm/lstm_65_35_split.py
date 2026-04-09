import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# load dataset
df = pd.read_csv("../../utrecht/processed_data/utrecht_pv_data.csv")

# process time
df["time"] = pd.to_datetime(df["time"])
df = df.set_index("time")
df = df.asfreq("h")

# add time features
df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)

# select features
features = [
    "pv_total_kWh",
    "solar_radiation_Wm2",
    "cloud_cover_okta",
    "temperature_C",
    "wind_speed_ms",
    "humidity_percent",
    "hour_sin",
    "hour_cos"
]

df = df[features]

# fill missing values
df = df.interpolate().bfill().ffill()

# split 65/35
split_idx = int(len(df) * 0.65)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

# scale
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_df)
test_scaled = scaler.transform(test_df)

# sequence creation
def create_sequences(data, seq_length=48):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0])
    return np.array(X), np.array(y)

SEQ_LEN = 48

X_train, y_train = create_sequences(train_scaled, SEQ_LEN)
X_test, y_test = create_sequences(test_scaled, SEQ_LEN)

# build model
model = Sequential([
    Input(shape=(SEQ_LEN, X_train.shape[2])),

    LSTM(128, return_sequences=True),
    Dropout(0.2),

    LSTM(64),
    Dropout(0.2),

    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

# train
model.fit(
    X_train, y_train,
    validation_split=0.1,
    epochs=30,
    batch_size=64,
    verbose=1
)

# predict
y_pred = model.predict(X_test)

# inverse scaling
y_test_rescaled = scaler.inverse_transform(
    np.concatenate([y_test.reshape(-1,1), np.zeros((len(y_test), len(features)-1))], axis=1)
)[:,0]

y_pred_rescaled = scaler.inverse_transform(
    np.concatenate([y_pred, np.zeros((len(y_pred), len(features)-1))], axis=1)
)[:,0]

# metrics
print("MAE:", mean_absolute_error(y_test_rescaled, y_pred_rescaled))
print("RMSE:", np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled)))