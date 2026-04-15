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

df = df.asfreq("h")

# target variable
y = df["pv_total_kWh"].clip(lower=0)

# optional stabilization (helps convergence)
y = np.log1p(y)

# select better exogenous variables (reduce multicollinearity)
exog = df[[
    "solar_radiation_Wm2",
    "cloud_cover_okta"
]]

# clean exog
exog = exog.replace([np.inf, -np.inf], np.nan)
exog = exog.interpolate().bfill().ffill()

# scale exog (important for convergence)
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
train_size = int(len(df) * 0.8)

y_train, y_test = y[:train_size], y[train_size:]
exog_train, exog_test = exog_scaled[:train_size], exog_scaled[train_size:]

# reduced grid search
p = q = range(0, 2)
d = [0]
P = Q = range(0, 2)
D = [0]
s = 24

pdq = list(itertools.product(p, d, q))
seasonal_pdq = list(itertools.product(P, D, Q))

best_aic = np.inf
best_order = None
best_seasonal_order = None

for order in pdq:
    for seasonal in seasonal_pdq:
        seasonal_order = (seasonal[0], seasonal[1], seasonal[2], s)
        try:
            model = SARIMAX(
                y_train,
                exog=exog_train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=True,
                enforce_invertibility=True
            )

            results = model.fit(method="lbfgs", maxiter=200, disp=False)

            if results.aic < best_aic:
                best_aic = results.aic
                best_order = order
                best_seasonal_order = seasonal_order

        except Exception:
            continue

print("best order:", best_order)
print("best seasonal order:", best_seasonal_order)
print("best AIC:", best_aic)

# fallback
if best_order is None:
    best_order = (1, 0, 1)
    best_seasonal_order = (1, 0, 1, 24)

# train final model
model = SARIMAX(
    y_train,
    exog=exog_train,
    order=best_order,
    seasonal_order=best_seasonal_order,
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

# convert back from log scale
y_test_real = np.expm1(y_test)
y_pred_real = np.expm1(y_pred)

# evaluation
mae = mean_absolute_error(y_test_real, y_pred_real)
rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))

print("MAE:", mae)
print("RMSE:", rmse)

# plot
plt.figure(figsize=(12, 6))
plt.plot(y_test_real.index, y_test_real, label="actual")
plt.plot(y_test_real.index, y_pred_real, label="predicted")
plt.legend()
plt.title("sarimax forecast (improved)")
plt.show()
