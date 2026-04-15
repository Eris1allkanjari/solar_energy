import pandas as pd
import numpy as np

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# load dataset
df = pd.read_csv("../../utrecht/processed_data/utrecht_pv_data.csv")

# process time
df["time"] = pd.to_datetime(df["time"])
df = df.set_index("time")
df = df.asfreq("h")

# target variable
y = df["pv_total_kWh"]

# handle missing values
y = y.interpolate().bfill().ffill()

# train/test split
train_size = int(len(y) * 0.8)
train, test = y[:train_size], y[train_size:]
test = test[:1000]

# store predictions
predictions = []

# rolling forecasting
history = list(train)

for t in range(len(test)):
    if t % 100 == 0:
        print(f"step {t}/{len(test)}")
    # fit model at each step (no seasonality)
    model = SARIMAX(
        history,
        order=(2, 0, 1),  # use the best ARIMA model
        seasonal_order=(0, 0, 0, 0),  # turn off periodicity
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    model_fit = model.fit(disp=False)

    # predict next step
    yhat = model_fit.forecast(steps=1)[0]
    predictions.append(yhat)

    # update history with actual value
    history.append(test.iloc[t])

# evaluation
mae = mean_absolute_error(test, predictions)
rmse = np.sqrt(mean_squared_error(test, predictions))

print("MAE:", mae)
print("RMSE:", rmse)