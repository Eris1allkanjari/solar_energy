import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import itertools

# load dataset
df = pd.read_csv("../../utrecht/processed_data/utrecht_pv_data.csv")

# timestamp processing
df["time"] = pd.to_datetime(df["time"])
df = df.set_index("time")

# target variable
y = df["pv_total_kWh"]

# handle solar zeros + clean NaNs properly
y = y.replace(0, np.nan)

# interpolate internal gaps
y = y.interpolate()

# fill edges
y = y.bfill().ffill()

print("NaNs in y after cleaning:", y.isna().sum())

# exogenous variables
exog = df[[
    "temperature_C",
    "wind_speed_ms",
    "humidity_percent",
    "cloud_cover_okta",
    "solar_radiation_Wm2"
]]

# clean exogenous variables (fixes your error)
exog = exog.replace([np.inf, -np.inf], np.nan)
exog = exog.interpolate()
exog = exog.bfill().ffill()

# align target and exogenous
y, exog = y.align(exog, join="inner")

# sanity check
print("NaNs in y:", y.isna().sum())
print("NaNs in exog:", exog.isna().sum().sum())

# check stationarity
result = adfuller(y)
print("ADF statistic:", result[0])
print("p-value:", result[1])

# plot acf and pacf
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plot_acf(y, lags=48, ax=plt.gca())

plt.subplot(1, 2, 2)
plot_pacf(y, lags=48, ax=plt.gca())

plt.tight_layout()
plt.show()

# train/test split
train_size = int(len(df) * 0.8)

y_train, y_test = y[:train_size], y[train_size:]
exog_train, exog_test = exog[:train_size], exog[train_size:]

# reduced grid search (since data is already stationary)
p = q = range(0, 2)
d = [0]
P = Q = range(0, 2)
D = [0]
s = 24  # daily seasonality

pdq = list(itertools.product(p, d, q))
seasonal_pdq = list(itertools.product(P, D, Q))

best_aic = np.inf
best_order = None
best_seasonal_order = None
errors = 0

# grid search
for order in pdq:
    for seasonal in seasonal_pdq:
        seasonal_order = (seasonal[0], seasonal[1], seasonal[2], s)
        try:
            model = SARIMAX(
                y_train,
                exog=exog_train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            results = model.fit(disp=False)

            if results.aic < best_aic:
                best_aic = results.aic
                best_order = order
                best_seasonal_order = seasonal_order

        except Exception:
            errors += 1
            continue

print("failed models:", errors)
print("best order:", best_order)
print("best seasonal order:", best_seasonal_order)
print("best AIC:", best_aic)

# fallback in case grid search fails
if best_order is None:
    print("using fallback model")
    best_order = (1, 0, 1)
    best_seasonal_order = (1, 0, 1, 24)

# train final model
model = SARIMAX(
    y_train,
    exog=exog_train,
    order=best_order,
    seasonal_order=best_seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False
)

results = model.fit()
print(results.summary())

# forecast
forecast = results.get_forecast(
    steps=len(y_test),
    exog=exog_test
)

y_pred = forecast.predicted_mean

# evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("MAE:", mae)
print("RMSE:", rmse)

# plot forecast
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label="actual")
plt.plot(y_test.index, y_pred, label="predicted")
plt.legend()
plt.title("sarimax forecast")
plt.show()

# residual diagnostics
residuals = results.resid

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(residuals)
plt.title("residuals")

plt.subplot(1, 2, 2)
plot_acf(residuals, lags=48, ax=plt.gca())

plt.tight_layout()
plt.show()
