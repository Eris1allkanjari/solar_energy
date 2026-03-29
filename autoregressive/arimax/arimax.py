import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import itertools

# load dataset
df = pd.read_csv("../../utrecht/processed_data/utrecht_pv_data.csv")

# timestamp processing
df["time"] = pd.to_datetime(df["time"])
df = df.set_index("time")

# set hourly frequency
df = df.asfreq("h")

# target variable
y = df["pv_total_kWh"].clip(lower=0)
y = y.interpolate().bfill().ffill()

# exogenous variables (same as your SARIMAX model)
exog = df[[
    "solar_radiation_Wm2",
    "cloud_cover_okta"
]]

# clean exog
exog = exog.replace([np.inf, -np.inf], np.nan)
exog = exog.interpolate().bfill().ffill()

# scale exog (important)
scaler = StandardScaler()
exog_scaled = pd.DataFrame(
    scaler.fit_transform(exog),
    index=exog.index,
    columns=exog.columns
)

# align data
y, exog_scaled = y.align(exog_scaled, join="inner")

# sanity check
print("NaNs in y:", y.isna().sum())
print("NaNs in exog:", exog_scaled.isna().sum().sum())

# stationarity check
result = adfuller(y)
print("ADF statistic:", result[0])
print("p-value:", result[1])

# train/test split
train_size = int(len(y) * 0.8)

y_train, y_test = y[:train_size], y[train_size:]
exog_train, exog_test = exog_scaled[:train_size], exog_scaled[train_size:]

# grid search (no seasonal part)
p = q = range(0, 3)
d = [0]  # already stationary

pdq = list(itertools.product(p, d, q))

best_aic = np.inf
best_order = None

# grid search
for order in pdq:
    try:
        model = SARIMAX(
            y_train,
            exog=exog_train,
            order=order,
            seasonal_order=(0, 0, 0, 0),  # ← disables seasonality
            enforce_stationarity=True,
            enforce_invertibility=True
        )

        results = model.fit(method="lbfgs", maxiter=200, disp=False)

        if results.aic < best_aic:
            best_aic = results.aic
            best_order = order

    except Exception:
        continue

print("best order:", best_order)
print("best AIC:", best_aic)

# fallback
if best_order is None:
    best_order = (1, 0, 1)

# train final model
model = SARIMAX(
    y_train,
    exog=exog_train,
    order=best_order,
    seasonal_order=(0, 0, 0, 0),
    enforce_stationarity=True,
    enforce_invertibility=True
)

results = model.fit(method="lbfgs", maxiter=300)
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

# plot
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label="actual")
plt.plot(y_test.index, y_pred, label="predicted")
plt.legend()
plt.title("arimax forecast")
plt.show()

# residual diagnostics
residuals = results.resid

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(residuals)
plt.title("residuals")

plt.subplot(1, 2, 2)
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(residuals, lags=48, ax=plt.gca())

plt.tight_layout()
plt.show()