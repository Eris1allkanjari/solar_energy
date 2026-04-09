import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# load dataset
df = pd.read_csv("../../utrecht/processed_data/utrecht_pv_data.csv")

# process time
df["time"] = pd.to_datetime(df["time"])
df = df.set_index("time")
df = df.asfreq("h")

# add time features
df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)

# add year/month for splitting
df["year"] = df.index.year
df["month"] = df.index.month

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

df = df[features + ["year", "month"]]

# fill missing values
df = df.interpolate().bfill().ffill()

# split by years
years = sorted(df["year"].unique())

train_df = df[df["year"].isin(years[:2])]
year3_df = df[df["year"] == years[2]]

val_idx, test_idx = [], []

for m in year3_df["month"].unique():
    month_data = year3_df[year3_df["month"] == m]
    split = int(len(month_data) * 0.3)
    val_idx.extend(month_data.index[:split])
    test_idx.extend(month_data.index[split:])

val_df = df.loc[val_idx]
test_df = df.loc[test_idx]

# drop helper columns
train_df = train_df[features]
val_df = val_df[features]
test_df = test_df[features]

# scale
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_df)
val_scaled = scaler.transform(val_df)
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
X_val, y_val = create_sequences(val_scaled, SEQ_LEN)
X_test, y_test = create_sequences(test_scaled, SEQ_LEN)

# build model
model = Sequential([
    Input(shape=(SEQ_LEN, X_train.shape[2])),

    GRU(128, return_sequences=True),
    Dropout(0.2),

    GRU(64),
    Dropout(0.2),

    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

# train
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=64,
    callbacks=[early_stop],
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