<<<<<<< HEAD
# P701 — PJM-West Power Load Forecast Dashboard

A live-feeling Streamlit dashboard built on top of your `P701_Model_Building_5_Model_Comparison_ARIMA_SARIMA_LSTM_XGBoost_RNN.ipynb` notebook.

## What's inside

```
app.py                  <- the Streamlit dashboard (run this)
build_artifacts.py      <- one-time offline script that reproduces the notebook
                            (feature engineering + trains all 5 models) and
                            writes everything the dashboard needs into artifacts/
requirements.txt        <- pip dependencies for running the dashboard
artifacts/
  processed_data.parquet    full engineered history (source for all charts/filters)
  xgb_model.json            trained, tuned XGBoost model (native booster format)
  xgb_feature_cols.json     exact column order XGBoost expects
  xgb_best_params.json      tuned hyperparameters (RandomizedSearchCV winners)
  holiday_dates.json        US federal holidays used for the Is_Holiday feature
  model_comparison.csv      MAE / RMSE / R2 for all 5 models on the 30-day eval window
  eval_predictions.csv      actual vs. all 5 models' predictions, hour-by-hour
  forecast_30d.csv          30-day forward forecast from XGBoost (best model)
  lstm_forecast_30d.csv     30-day forward forecast from LSTM (for comparison)
  rnn_forecast_30d.csv      30-day forward forecast from RNN (for comparison)
  meta.json                 best model name, data date range, eval window dates
```

## Why XGBoost is the "live" model

Your notebook's 30-day common evaluation showed XGBoost (Tuned) dramatically
outperforming the other four:

| Model | MAE | RMSE | R2 |
|---|---|---|---|
| **XGBoost (Tuned)** | **97.0** | **125.4** | **0.986** |
| RNN | 898.8 | 1065.8 | -0.038 |
| LSTM | 1130.1 | 1414.7 | -0.829 |
| ARIMA | 1181.6 | 1473.5 | -0.985 |
| SARIMA | 2805.5 | 3066.3 | -7.594 |

So the dashboard runs XGBoost live (it retrains a fresh recursive forecast in
under a second for any horizon you pick), and shows LSTM/RNN as precomputed
comparison lines — retraining a neural net on every slider tick isn't practical
in a live UI. ARIMA/SARIMA are also available in the "Model Comparison" tab.

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL it prints (usually `http://localhost:8501`).

## Dashboard features

- **Live tick mode** (sidebar toggle) — auto-refreshes the page every few
  seconds with a pulsing "LIVE" badge and a scrolling ticker tape of
  current load / best model / 24h stats, to simulate a streaming meter feed.
- **Adjustable time features** — filter and re-aggregate history by Hour,
  Day, Week, or Month; a date-range picker; hour-of-day slider; day-of-week
  multiselect; season filter; weekday/weekend toggle.
- **Model Comparison tab** — the 5-model leaderboard, RMSE bar chart, and an
  overlay of actual vs. every model's prediction on the shared 30-day hold-out.
- **Forecast tab** — pick a model (XGBoost/LSTM/RNN) and horizon (24h to 30
  days); XGBoost forecasts are generated live via the same recursive
  lag/rolling-feature logic as the notebook's final forecast cell.
- **Load Patterns tab** — average load by hour and month, an hour × weekday
  heatmap, and holiday vs. non-holiday comparison.
- **Raw Data tab** — the filtered slice of engineered data, downloadable as CSV.
- KPI cards and a CSS-animated ticker tape for a "trading dashboard" feel,
  built with Plotly dark-themed charts throughout.

## Regenerating artifacts (e.g. with new/updated data)

**You don't need this step to run the dashboard** — `artifacts/` already
ships with everything precomputed from the notebook's data. Only do this if
you have new/updated hourly load data you want to retrain on.

1. Put the raw `PJMW_MW_Hourly*.xlsx` file in the **same folder as
   `build_artifacts.py`** (the script auto-detects it there — no path editing
   needed).
2. Install dependencies, including TensorFlow (only needed for this offline
   rebuild step — see the platform notes below):
   ```bash
   pip install -r requirements.txt
   pip install tensorflow        # see notes below if this fails
   python3 build_artifacts.py
   ```

This re-runs the full notebook pipeline (feature engineering, XGBoost tuning,
ARIMA/SARIMA/LSTM/RNN training, 30-day evaluation, and 30-day forecasts) and
overwrites everything in `artifacts/`. It takes a few minutes — most of it is
the XGBoost `RandomizedSearchCV` and the two neural nets.

Note: `app.py` itself never imports TensorFlow — LSTM/RNN forecasts are
served from the precomputed CSVs, so the deployed dashboard stays light.

### TensorFlow install notes (only relevant for this optional rebuild step)

`tensorflow-cpu` doesn't publish wheels for every Python version/platform
combo (notably: newer Python versions like 3.13, and Apple Silicon Macs).
If `pip install tensorflow-cpu` fails:

- **Apple Silicon Mac (M1/M2/M3/M4)**: use `pip install tensorflow` (the
  regular package, not `-cpu`) — it has native Apple Silicon wheels.
- **Any platform, Python version issue**: TensorFlow wheels usually lag
  behind the newest Python release. Use a **Python 3.10 or 3.11** virtual
  environment just for this rebuild step:
  ```bash
  python3.11 -m venv tf_env
  source tf_env/bin/activate      # Windows: tf_env\Scripts\activate
  pip install -r requirements.txt
  pip install tensorflow
  python3 build_artifacts.py
  deactivate
  ```
- Once `artifacts/` is regenerated, you're done with that environment — the
  dashboard itself (`app.py`) runs fine on your normal Python setup.

## Deploying

**Streamlit Community Cloud** (free, easiest):
1. Push this folder to a GitHub repo (include `artifacts/`, minus the
   `.parquet` if your repo has size limits — see note below).
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo,
   set the main file to `app.py`.
3. Done — it builds from `requirements.txt` automatically.

**Note on `processed_data.parquet` size**: it's ~7MB for ~143K hourly rows,
fine for GitHub/Streamlit Cloud. If your dataset grows much larger, consider
trimming history kept in the parquet (e.g., last N years) since the dashboard
only really needs enough history for the filters/charts to be meaningful.

**Other options**: any host that runs a long-lived Python process works —
Render, Railway, an EC2/VM with `streamlit run app.py --server.port 80`, or
Hugging Face Spaces (Streamlit SDK).
=======
# Forecasting-Power-supply
>>>>>>> c4527814ee2011e7bf5c8354a5d61f8bd8f97879
