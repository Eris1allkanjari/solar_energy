import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# load dataset
df = pd.read_csv("../../utrecht/processed_data/utrecht_pv_data.csv")

# convert time column and set index
df["time"] = pd.to_datetime(df["time"])
df = df.set_index("time")

# enforce hourly frequency
df = df.asfreq("h")

# adding time features (critical for solar)
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

# scale data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)


# create sequences
def create_sequences(data, seq_length=48):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)


# use longer memory
SEQ_LEN = 48

X, y = create_sequences(data_scaled, SEQ_LEN)

# split into train and test
train_size = int(len(X) * 0.8)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# build stronger LSTM model
model = Sequential([
    Input(shape=(SEQ_LEN, X.shape[2])),  # ← fix here

    LSTM(128, return_sequences=True),
    Dropout(0.2),

    LSTM(64),
    Dropout(0.2),

    Dense(32, activation="relu"),
    Dense(1)
])

# compile model
model.compile(optimizer="adam", loss="mse")

# train model
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# generate predictions
y_pred = model.predict(X_test)

# inverse scale true values
y_test_rescaled = scaler.inverse_transform(
    np.concatenate([y_test.reshape(-1, 1), np.zeros((len(y_test), len(features) - 1))], axis=1)
)[:, 0]

# inverse scale predictions
y_pred_rescaled = scaler.inverse_transform(
    np.concatenate([y_pred, np.zeros((len(y_pred), len(features) - 1))], axis=1)
)[:, 0]

# compute metrics
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))

print("MAE:", mae)
print("RMSE:", rmse)

# plot results
plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled, label="actual")
plt.plot(y_pred_rescaled, label="predicted")
plt.legend()
plt.title("improved lstm forecast")
plt.show()