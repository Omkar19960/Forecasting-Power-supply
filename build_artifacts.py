"""
Builds every artifact the Streamlit dashboard needs, using the exact
feature engineering / modeling logic from
P701_Model_Building_5_Model_Comparison_ARIMA_SARIMA_LSTM_XGBoost_RNN.ipynb

Run once (offline). Produces, inside ./artifacts/:
    processed_data.parquet     -> full engineered dataframe (for the dashboard's history views)
    xgb_model.json             -> tuned XGBoost model (native booster save)
    xgb_feature_cols.json      -> exact column order XGBoost expects
    xgb_best_params.json       -> tuned hyperparameters (re-used for live retrains)
    holiday_dates.json         -> US federal holiday set used for Is_Holiday feature
    model_comparison.csv       -> 30-day MAE/RMSE/R2 comparison across all 5 models
    eval_predictions.csv       -> the 30-day evaluation window: actual + all 5 model predictions
    forecast_30d.csv           -> next-30-day forecast from the best model (XGBoost)
    lstm_forecast_30d.csv      -> next-30-day forecast from LSTM (for comparison toggle)
    rnn_forecast_30d.csv       -> next-30-day forecast from RNN (for comparison toggle)
    meta.json                  -> misc info (best model name, data date range, etc.)
"""
import json
import os
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, SimpleRNN, Dense
from tensorflow.keras.callbacks import EarlyStopping
tf.random.set_seed(42)

ART = "artifacts"
os.makedirs(ART, exist_ok=True)

import glob

# Looks for the raw data file next to this script first (portable), falling back
# to a couple of common names so this runs the same on any machine.
_candidates = (
    glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "PJMW_MW_Hourly*.xlsx"))
    or glob.glob("PJMW_MW_Hourly*.xlsx")
)
if not _candidates:
    raise FileNotFoundError(
        "Could not find PJMW_MW_Hourly*.xlsx.\n"
        "Place the raw data file in the SAME folder as build_artifacts.py "
        "(next to this script), then run it again.\n"
        f"Looked in: {os.path.dirname(os.path.abspath(__file__))} and the current working directory."
    )
DATA_PATH = _candidates[0]
print(f"Using data file: {DATA_PATH}")

# ---------------------------------------------------------------- 1. Load & clean
print("Loading data...")
df = pd.read_excel(DATA_PATH)
df = df.rename(columns={"Datetime": "Timestamp", "PJMW_MW": "Power_Load"})
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)
df = df.drop_duplicates(subset="Timestamp", keep="first").reset_index(drop=True)
print("Rows:", len(df))

# ---------------------------------------------------------------- 2. Feature engineering
df["Hour_Slot"] = df["Timestamp"].dt.hour
df["Weekday_Number"] = df["Timestamp"].dt.dayofweek
df["Date_Number"] = df["Timestamp"].dt.day
df["Month_Index"] = df["Timestamp"].dt.month
df["Calendar_Year"] = df["Timestamp"].dt.year
df["Weekend_Flag"] = (df["Weekday_Number"] >= 5).astype(int)

season_map = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn"
}
df["Season_Group"] = df["Month_Index"].map(season_map)

cal = USFederalHolidayCalendar()
holiday_index = cal.holidays(start=df["Timestamp"].min().normalize(), end=df["Timestamp"].max().normalize())
holiday_dates = set(pd.DatetimeIndex(holiday_index).date)
df["Is_Holiday"] = df["Timestamp"].dt.date.isin(holiday_dates).astype(int)

for h in [1, 2, 3, 24, 48, 168]:
    df[f"Load_Lag_{h}H"] = df["Power_Load"].shift(h)

df["Rolling_Mean_24H"] = df["Power_Load"].shift(1).rolling(24).mean()
df["Rolling_Mean_7D"] = df["Power_Load"].shift(1).rolling(168).mean()
df["Rolling_Std_24H"] = df["Power_Load"].shift(1).rolling(24).std()
df["Rolling_Std_7D"] = df["Power_Load"].shift(1).rolling(168).std()

df.to_parquet(f"{ART}/processed_data.parquet", index=False)
with open(f"{ART}/holiday_dates.json", "w") as f:
    json.dump(sorted(d.isoformat() for d in holiday_dates), f)
print("Processed data saved.")

# ---------------------------------------------------------------- 3. Model matrix
feature_cols = [
    "Hour_Slot", "Weekday_Number", "Date_Number", "Month_Index", "Calendar_Year",
    "Weekend_Flag", "Is_Holiday",
    "Load_Lag_1H", "Load_Lag_2H", "Load_Lag_3H", "Load_Lag_24H", "Load_Lag_48H", "Load_Lag_168H",
    "Rolling_Mean_24H", "Rolling_Mean_7D", "Rolling_Std_24H", "Rolling_Std_7D"
]
model_df = df[["Timestamp", "Power_Load"] + feature_cols + ["Season_Group"]].dropna().copy()
model_df = pd.get_dummies(model_df, columns=["Season_Group"], drop_first=False)

# ---------------------------------------------------------------- 4. Train/test split (last 1 year held out)
test_start = model_df["Timestamp"].max() - pd.DateOffset(years=1)
train = model_df[model_df["Timestamp"] < test_start]
test = model_df[model_df["Timestamp"] >= test_start]

X_train = train.drop(columns=["Timestamp", "Power_Load"])
y_train = train["Power_Load"]
X_test = test.drop(columns=["Timestamp", "Power_Load"]).reindex(columns=X_train.columns, fill_value=0)
y_test = test["Power_Load"]
print("Train:", X_train.shape, "Test:", X_test.shape)

# ---------------------------------------------------------------- 5. Tune XGBoost
print("Tuning XGBoost (RandomizedSearchCV, this takes a bit)...")
tscv = TimeSeriesSplit(n_splits=3)
xgb_param_grid = {
    "n_estimators": [100, 200, 300, 500],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "max_depth": [3, 4, 5, 6, 8],
    "min_child_weight": [1, 3, 5],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "gamma": [0, 0.1, 0.2]
}
xgb_search = RandomizedSearchCV(
    XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1),
    param_distributions=xgb_param_grid,
    n_iter=15, cv=tscv, scoring="neg_root_mean_squared_error",
    random_state=42, n_jobs=-1
)
xgb_search.fit(X_train, y_train)
print("Best XGBoost params:", xgb_search.best_params_)
with open(f"{ART}/xgb_best_params.json", "w") as f:
    json.dump(xgb_search.best_params_, f)

# ---------------------------------------------------------------- 6. Common 30-day evaluation window
eval_hours = 24 * 30
eval_start = test["Timestamp"].max() - pd.Timedelta(hours=eval_hours - 1)
eval_test = test[test["Timestamp"] >= eval_start].copy().reset_index(drop=True)

train_end = eval_test["Timestamp"].min() - pd.Timedelta(hours=1)
train_start = train_end - pd.Timedelta(hours=24 * 90 - 1)
comparison_train = df[(df["Timestamp"] >= train_start) & (df["Timestamp"] <= train_end)].copy()
comparison_model_df = model_df[(model_df["Timestamp"] >= train_start) & (model_df["Timestamp"] <= train_end)].copy()
print("Eval window:", eval_test["Timestamp"].min(), "->", eval_test["Timestamp"].max())

# ---------------------------------------------------------------- 7. XGBoost (comparison window)
xgb_compare = XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1, **xgb_search.best_params_)
xgb_feature_cols = X_train.columns.tolist()
with open(f"{ART}/xgb_feature_cols.json", "w") as f:
    json.dump(xgb_feature_cols, f)

X_cmp_train = comparison_model_df.drop(columns=["Timestamp", "Power_Load"]).reindex(columns=xgb_feature_cols, fill_value=0)
y_cmp_train = comparison_model_df["Power_Load"]
X_cmp_test = eval_test.drop(columns=["Timestamp", "Power_Load"]).reindex(columns=xgb_feature_cols, fill_value=0)
y_eval = eval_test["Power_Load"].values

xgb_compare.fit(X_cmp_train, y_cmp_train)
pred_xgb_tuned = xgb_compare.predict(X_cmp_test)
print("XGBoost comparison done.")

# ---------------------------------------------------------------- 8. ARIMA
arima_series = comparison_train.set_index("Timestamp")["Power_Load"].asfreq("h").interpolate(limit_direction="both")
print("Fitting ARIMA...")
arima_model = ARIMA(arima_series, order=(1, 1, 1)).fit()
pred_arima = np.asarray(arima_model.forecast(steps=len(eval_test)))

# ---------------------------------------------------------------- 9. SARIMA
print("Fitting SARIMA...")
sarima_model = SARIMAX(arima_series, order=(1, 1, 1), seasonal_order=(1, 0, 1, 24),
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
pred_sarima = np.asarray(sarima_model.forecast(steps=len(eval_test)))

# ---------------------------------------------------------------- 10. LSTM & RNN
def make_sequences(values, seq_len=24):
    X, y = [], []
    for i in range(seq_len, len(values)):
        X.append(values[i - seq_len:i, 0])
        y.append(values[i, 0])
    return np.array(X).reshape(-1, seq_len, 1), np.array(y)

seq_len = 24
series_values = comparison_train["Power_Load"].astype(float).values.reshape(-1, 1)
scaler_nn = MinMaxScaler()
scaled = scaler_nn.fit_transform(series_values)
X_seq, y_seq = make_sequences(scaled, seq_len)

print("Training LSTM...")
tf.keras.backend.clear_session()
lstm_model = Sequential([LSTM(32, input_shape=(seq_len, 1)), Dense(16, activation="relu"), Dense(1)])
lstm_model.compile(optimizer="adam", loss="mse")
early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
lstm_model.fit(X_seq, y_seq, epochs=10, batch_size=32, validation_split=0.1, callbacks=[early_stop], verbose=0, shuffle=False)

window = scaled[-seq_len:, 0].tolist()
pred_scaled = []
for _ in range(len(eval_test)):
    x_input = np.array(window[-seq_len:]).reshape(1, seq_len, 1)
    next_scaled = float(lstm_model.predict(x_input, verbose=0)[0, 0])
    pred_scaled.append(next_scaled)
    window.append(next_scaled)
pred_lstm = scaler_nn.inverse_transform(np.array(pred_scaled).reshape(-1, 1)).ravel()

print("Training RNN...")
tf.keras.backend.clear_session()
rnn_model = Sequential([SimpleRNN(32, input_shape=(seq_len, 1)), Dense(16, activation="relu"), Dense(1)])
rnn_model.compile(optimizer="adam", loss="mse")
early_stop_rnn = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
rnn_model.fit(X_seq, y_seq, epochs=10, batch_size=32, validation_split=0.1, callbacks=[early_stop_rnn], verbose=0, shuffle=False)

window = scaled[-seq_len:, 0].tolist()
pred_scaled_rnn = []
for _ in range(len(eval_test)):
    x_input = np.array(window[-seq_len:]).reshape(1, seq_len, 1)
    next_scaled = float(rnn_model.predict(x_input, verbose=0)[0, 0])
    pred_scaled_rnn.append(next_scaled)
    window.append(next_scaled)
pred_rnn = scaler_nn.inverse_transform(np.array(pred_scaled_rnn).reshape(-1, 1)).ravel()

# ---------------------------------------------------------------- 11. Comparison table
def calculate_metrics(y_true, y_pred):
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": r2_score(y_true, y_pred)}

model_predictions = {
    "XGBoost (Tuned)": pred_xgb_tuned,
    "ARIMA": pred_arima,
    "SARIMA": pred_sarima,
    "LSTM": pred_lstm,
    "RNN": pred_rnn,
}
comparison_rows = [{"Model": name, **calculate_metrics(y_eval, pred)} for name, pred in model_predictions.items()]
final_comparison = pd.DataFrame(comparison_rows).sort_values("RMSE").reset_index(drop=True)
final_comparison.to_csv(f"{ART}/model_comparison.csv", index=False)
print(final_comparison)

eval_out = eval_test[["Timestamp"]].copy()
eval_out["Actual"] = y_eval
for name, pred in model_predictions.items():
    eval_out[name] = pred
eval_out.to_csv(f"{ART}/eval_predictions.csv", index=False)

best_model_name = final_comparison.loc[0, "Model"]
print("Best model:", best_model_name)

# ---------------------------------------------------------------- 12. Final XGBoost fit on ALL data (for live dashboard use)
X_all = model_df.drop(columns=["Timestamp", "Power_Load"])
y_all = model_df["Power_Load"]
final_xgb = XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1, **xgb_search.best_params_)
final_xgb.fit(X_all, y_all)
final_xgb.save_model(f"{ART}/xgb_model.json")
print("Final XGBoost model saved.")

# ---------------------------------------------------------------- 13. 30-day forward forecast: XGBoost (recursive)
def xgb_recursive_forecast(model, model_df, expected_cols, holiday_dates, steps=720):
    history = model_df[["Timestamp", "Power_Load"]].copy()
    future_rows = []
    for _ in range(steps):
        next_ts = history["Timestamp"].max() + pd.Timedelta(hours=1)
        recent_values = history["Power_Load"].tolist()
        row = {
            "Timestamp": next_ts, "Hour_Slot": next_ts.hour,
            "Weekday_Number": next_ts.dayofweek, "Date_Number": next_ts.day,
            "Month_Index": next_ts.month, "Calendar_Year": next_ts.year,
            "Weekend_Flag": int(next_ts.dayofweek >= 5),
            "Is_Holiday": int(next_ts.normalize().date() in holiday_dates),
            "Season_Group_Spring": int(next_ts.month in [3, 4, 5]),
            "Season_Group_Summer": int(next_ts.month in [6, 7, 8]),
            "Season_Group_Autumn": int(next_ts.month in [9, 10, 11]),
            "Season_Group_Winter": int(next_ts.month in [12, 1, 2]),
        }
        for lag in [1, 2, 3, 24, 48, 168]:
            row[f"Load_Lag_{lag}H"] = recent_values[-lag]
        row["Rolling_Mean_24H"] = np.mean(recent_values[-24:])
        row["Rolling_Mean_7D"] = np.mean(recent_values[-168:])
        row["Rolling_Std_24H"] = np.std(recent_values[-24:], ddof=1)
        row["Rolling_Std_7D"] = np.std(recent_values[-168:], ddof=1)
        X_future = pd.DataFrame([row]).reindex(columns=expected_cols, fill_value=0)
        next_pred = float(model.predict(X_future)[0])
        history = pd.concat([history, pd.DataFrame([{"Timestamp": next_ts, "Power_Load": next_pred}])], ignore_index=True)
        future_rows.append({"Timestamp": next_ts, "Forecast_Power_Load": next_pred})
    return pd.DataFrame(future_rows)

print("Generating 30-day XGBoost forecast...")
expected_cols = list(X_all.columns)
forecast_30d = xgb_recursive_forecast(final_xgb, model_df, expected_cols, holiday_dates, steps=720)
forecast_30d.to_csv(f"{ART}/forecast_30d.csv", index=False)

# ---------------------------------------------------------------- 14. 30-day forward forecast: LSTM & RNN (for toggle/compare)
recent_90d = df[df["Timestamp"] >= df["Timestamp"].max() - pd.Timedelta(days=90)].copy()
series_final = recent_90d.set_index("Timestamp")["Power_Load"].asfreq("h").interpolate(limit_direction="both")
scaler_final = MinMaxScaler()
scaled_final = scaler_final.fit_transform(series_final.values.reshape(-1, 1))
Xf, yf = make_sequences(scaled_final, 24)
future_start = df["Timestamp"].max() + pd.Timedelta(hours=1)
future_index = pd.date_range(future_start, periods=720, freq="h")

for label, layer in [("lstm", LSTM), ("rnn", SimpleRNN)]:
    print(f"Training final {label.upper()} for 30-day forecast...")
    tf.keras.backend.clear_session()
    m = Sequential([layer(32, input_shape=(24, 1)), Dense(16, activation="relu"), Dense(1)])
    m.compile(optimizer="adam", loss="mse")
    m.fit(Xf, yf, epochs=10, batch_size=32, validation_split=0.1, verbose=0, shuffle=False)
    window = scaled_final[-24:, 0].tolist()
    preds = []
    for _ in range(720):
        x_in = np.array(window[-24:]).reshape(1, 24, 1)
        nxt = float(m.predict(x_in, verbose=0)[0, 0])
        preds.append(nxt)
        window.append(nxt)
    preds = scaler_final.inverse_transform(np.array(preds).reshape(-1, 1)).ravel()
    pd.DataFrame({"Timestamp": future_index, "Forecast_Power_Load": preds}).to_csv(f"{ART}/{label}_forecast_30d.csv", index=False)

# ---------------------------------------------------------------- 15. meta.json
meta = {
    "best_model_name": best_model_name,
    "data_min_ts": df["Timestamp"].min().isoformat(),
    "data_max_ts": df["Timestamp"].max().isoformat(),
    "n_rows": int(len(df)),
    "eval_window_start": eval_test["Timestamp"].min().isoformat(),
    "eval_window_end": eval_test["Timestamp"].max().isoformat(),
}
with open(f"{ART}/meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nALL ARTIFACTS BUILT SUCCESSFULLY.")
