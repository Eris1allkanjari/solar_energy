import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# load dataset
df = pd.read_csv("../../utrecht/processed_data/utrecht_pv_data.csv")

# convert time column and set index
df["time"] = pd.to_datetime(df["time"])
df = df.set_index("time")

# enforce hourly frequency
df = df.asfreq("h")

# select features
features = [
    "pv_total_kWh",
    "solar_radiation_Wm2",
    "cloud_cover_okta",
    "temperature_C",
    "wind_speed_ms",
    "humidity_percent"
]

df = df[features]

# fill missing values
df = df.interpolate().bfill().ffill()

# scale data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)

# create sequences
def create_sequences(data, seq_length=24):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0])
    return np.array(X), np.array(y)

# define sequence length
SEQ_LEN = 24

X, y = create_sequences(data_scaled, SEQ_LEN)

# split into train and test
train_size = int(len(X) * 0.8)

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# build LSTM model
model = Sequential([
    LSTM(64, activation="tanh", return_sequences=False, input_shape=(SEQ_LEN, X.shape[2])),
    Dense(32, activation="relu"),
    Dense(1)
])

# compile model
model.compile(optimizer="adam", loss="mse")

# train model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# generate predictions
y_pred = model.predict(X_test)

# inverse scale true values
y_test_rescaled = scaler.inverse_transform(
    np.concatenate([y_test.reshape(-1,1), np.zeros((len(y_test), len(features)-1))], axis=1)
)[:,0]

# inverse scale predictions
y_pred_rescaled = scaler.inverse_transform(
    np.concatenate([y_pred, np.zeros((len(y_pred), len(features)-1))], axis=1)
)[:,0]

# compute metrics
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))

print("MAE:", mae)
print("RMSE:", rmse)

# plot results
plt.figure(figsize=(12,6))
plt.plot(y_test_rescaled, label="actual")
plt.plot(y_pred_rescaled, label="predicted")
plt.legend()
plt.title("lstm forecast")
plt.show()