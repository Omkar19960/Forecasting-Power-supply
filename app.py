"""
P701 Power Load Forecasting — Live Dashboard
Models: XGBoost (Tuned) selected as production model, benchmarked against
ARIMA, SARIMA, LSTM, RNN.

Run:
    streamlit run app.py
"""
import json
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from xgboost import XGBRegressor

# ------------------------------------------------------------------ PAGE CONFIG
st.set_page_config(
    page_title="PJM-West Power Load Forecast",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

ART = "artifacts"

# ------------------------------------------------------------------ THEME / CSS
st.markdown("""
<style>
@keyframes pulse {
  0%   { opacity: 1; }
  50%  { opacity: 0.55; }
  100% { opacity: 1; }
}
@keyframes ticker {
  0%   { transform: translateX(100%); }
  100% { transform: translateX(-100%); }
}
.live-dot {
  height: 10px; width: 10px; border-radius: 50%;
  background-color: #00e676; display: inline-block;
  margin-right: 8px; animation: pulse 1.4s infinite;
  box-shadow: 0 0 8px #00e676;
}
.live-badge {
  font-size: 0.85rem; color: #00e676; font-weight: 600;
  letter-spacing: 0.5px; text-transform: uppercase;
}
.ticker-wrap {
  width: 100%; overflow: hidden; background: #0d1117;
  border: 1px solid #21262d; border-radius: 8px;
  padding: 10px 0; margin-bottom: 14px;
}
.ticker-move {
  display: inline-block; white-space: nowrap;
  animation: ticker 22s linear infinite;
  font-family: 'Courier New', monospace; font-size: 0.95rem;
}
.metric-card {
  background: linear-gradient(145deg, #12161c, #1a1f27);
  border: 1px solid #262c36; border-radius: 12px;
  padding: 16px 18px; text-align: left;
}
.metric-card h3 { margin: 0; font-size: 0.78rem; color: #8b949e; font-weight: 500; text-transform: uppercase; letter-spacing: .5px;}
.metric-card p { margin: 4px 0 0 0; font-size: 1.6rem; font-weight: 700; color: #e6edf3; }
.up   { color: #3fb950 !important; }
.down { color: #f85149 !important; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ CACHED LOADERS
@st.cache_data(show_spinner=False)
def load_processed_data():
    df = pd.read_parquet(f"{ART}/processed_data.parquet")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


@st.cache_data(show_spinner=False)
def load_meta():
    with open(f"{ART}/meta.json") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_holiday_dates():
    with open(f"{ART}/holiday_dates.json") as f:
        raw = json.load(f)
    return set(pd.to_datetime(raw).date)


@st.cache_data(show_spinner=False)
def load_comparison():
    return pd.read_csv(f"{ART}/model_comparison.csv")


@st.cache_data(show_spinner=False)
def load_eval_predictions():
    df = pd.read_csv(f"{ART}/eval_predictions.csv")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


@st.cache_data(show_spinner=False)
def load_forecast(name):
    fname = {"XGBoost (Tuned)": "forecast_30d.csv", "LSTM": "lstm_forecast_30d.csv", "RNN": "rnn_forecast_30d.csv"}[name]
    df = pd.read_csv(f"{ART}/{fname}")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


@st.cache_resource(show_spinner=False)
def load_xgb_model():
    model = XGBRegressor()
    model.load_model(f"{ART}/xgb_model.json")
    with open(f"{ART}/xgb_feature_cols.json") as f:
        cols = json.load(f)
    return model, cols


# ------------------------------------------------------------------ LIVE FORECAST HELPER (recursive, cached per horizon)
@st.cache_data(show_spinner=False)
def recursive_forecast(steps, _model_tuple):
    model, expected_cols = _model_tuple
    df = load_processed_data()
    holiday_dates = load_holiday_dates()
    feature_cols_base = [
        "Hour_Slot", "Weekday_Number", "Date_Number", "Month_Index", "Calendar_Year",
        "Weekend_Flag", "Is_Holiday",
        "Load_Lag_1H", "Load_Lag_2H", "Load_Lag_3H", "Load_Lag_24H", "Load_Lag_48H", "Load_Lag_168H",
        "Rolling_Mean_24H", "Rolling_Mean_7D", "Rolling_Std_24H", "Rolling_Std_7D"
    ]
    season_map = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
                  6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}
    model_df = df[["Timestamp", "Power_Load"] + feature_cols_base + ["Season_Group"]].dropna().copy()
    model_df = pd.get_dummies(model_df, columns=["Season_Group"], drop_first=False)

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


# ------------------------------------------------------------------ LOAD EVERYTHING
df = load_processed_data()
meta = load_meta()
comparison = load_comparison()
eval_preds = load_eval_predictions()
xgb_model, xgb_cols = load_xgb_model()

MODEL_COLORS = {
    "Actual": "#e6edf3",
    "XGBoost (Tuned)": "#00e676",
    "ARIMA": "#58a6ff",
    "SARIMA": "#f778ba",
    "LSTM": "#ffa657",
    "RNN": "#d2a8ff",
}

# ------------------------------------------------------------------ SIDEBAR — FILTERS
st.sidebar.markdown("## ⚡ Control Panel")
st.sidebar.caption("Filter the historical view and configure the live forecast.")

st.sidebar.markdown("#### 📅 Time Granularity")
granularity = st.sidebar.radio(
    "Aggregate history by:",
    ["Hour", "Day", "Week", "Month"],
    horizontal=True,
)

st.sidebar.markdown("#### 🗓️ Date Range")
min_d, max_d = df["Timestamp"].min().date(), df["Timestamp"].max().date()
default_start = max_d - timedelta(days=180)
date_range = st.sidebar.date_input(
    "History window", value=(max(default_start, min_d), max_d),
    min_value=min_d, max_value=max_d,
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_d, max_d

st.sidebar.markdown("#### ⏰ Hour-of-Day Filter")
hour_range = st.sidebar.slider("Hour range", 0, 23, (0, 23))

st.sidebar.markdown("#### 📆 Day-of-Week Filter")
dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
selected_dows = st.sidebar.multiselect("Weekdays", dow_labels, default=dow_labels)

st.sidebar.markdown("#### 🍂 Season Filter")
seasons = st.sidebar.multiselect("Seasons", ["Winter", "Spring", "Summer", "Autumn"],
                                  default=["Winter", "Spring", "Summer", "Autumn"])

only_weekend = st.sidebar.selectbox("Day type", ["All days", "Weekdays only", "Weekends only"])

st.sidebar.markdown("---")
st.sidebar.markdown("#### 🔮 Forecast Settings")
forecast_model = st.sidebar.selectbox(
    "Forecast model", ["XGBoost (Tuned)", "LSTM", "RNN"],
    help="XGBoost is the production-selected model (lowest RMSE). LSTM/RNN shown for comparison."
)
horizon_choice = st.sidebar.select_slider(
    "Forecast horizon", options=["24 hours", "3 days", "7 days", "14 days", "30 days"], value="7 days"
)
horizon_hours = {"24 hours": 24, "3 days": 72, "7 days": 168, "14 days": 336, "30 days": 720}[horizon_choice]

st.sidebar.markdown("---")
live_mode = st.sidebar.toggle("🔴 Live tick mode", value=False,
                               help="Simulates a streaming meter reading with auto-refresh.")
if live_mode:
    refresh_secs = st.sidebar.slider("Refresh interval (sec)", 2, 10, 3)

# ------------------------------------------------------------------ APPLY FILTERS TO HISTORY
mask = (
    (df["Timestamp"].dt.date >= start_date) & (df["Timestamp"].dt.date <= end_date) &
    (df["Hour_Slot"].between(hour_range[0], hour_range[1])) &
    (df["Weekday_Number"].isin([dow_labels.index(d) for d in selected_dows])) &
    (df["Season_Group"].isin(seasons))
)
if only_weekend == "Weekdays only":
    mask &= df["Weekend_Flag"] == 0
elif only_weekend == "Weekends only":
    mask &= df["Weekend_Flag"] == 1

filtered = df.loc[mask].copy()

# ------------------------------------------------------------------ AGGREGATION
agg_rule = {"Hour": "h", "Day": "D", "Week": "W", "Month": "MS"}[granularity]
if not filtered.empty:
    agg_df = (
        filtered.set_index("Timestamp")["Power_Load"]
        .resample(agg_rule).mean().dropna().reset_index()
    )
else:
    agg_df = pd.DataFrame(columns=["Timestamp", "Power_Load"])

# ------------------------------------------------------------------ HEADER
top_l, top_r = st.columns([0.75, 0.25])
with top_l:
    st.markdown("# ⚡ PJM-West Power Load Forecast — Live Ops Dashboard")
    st.caption(f"Hourly demand (MW) · {meta['n_rows']:,} records · "
               f"{pd.to_datetime(meta['data_min_ts']).date()} → {pd.to_datetime(meta['data_max_ts']).date()}")
with top_r:
    st.markdown(
        f"<div style='text-align:right; padding-top: 18px;'>"
        f"<span class='live-dot'></span><span class='live-badge'>{'LIVE' if live_mode else 'STATIC'}</span>"
        f"</div>", unsafe_allow_html=True
    )

# ------------------------------------------------------------------ TICKER TAPE
last_24 = df.tail(24)
latest_val = df["Power_Load"].iloc[-1]
prev_val = df["Power_Load"].iloc[-2]
delta_pct = (latest_val - prev_val) / prev_val * 100
best_model_row = comparison.iloc[0]

ticker_items = [
    f"PJMW LATEST LOAD: {latest_val:,.0f} MW ({'▲' if delta_pct >= 0 else '▼'}{delta_pct:+.2f}%)",
    f"BEST MODEL: {best_model_row['Model']} — RMSE {best_model_row['RMSE']:.1f} MW",
    f"24H AVG: {last_24['Power_Load'].mean():,.0f} MW",
    f"24H PEAK: {last_24['Power_Load'].max():,.0f} MW",
    f"24H LOW: {last_24['Power_Load'].min():,.0f} MW",
    f"SEASON: {df['Season_Group'].iloc[-1]}",
    f"HOLIDAY TODAY: {'YES' if df['Is_Holiday'].iloc[-1] else 'NO'}",
]
ticker_text = "     •     ".join(ticker_items) * 2
st.markdown(f"<div class='ticker-wrap'><div class='ticker-move'>&nbsp;&nbsp;{ticker_text}</div></div>",
            unsafe_allow_html=True)

# ------------------------------------------------------------------ KPI CARDS
k1, k2, k3, k4, k5 = st.columns(5)
delta_class = "up" if delta_pct >= 0 else "down"
arrow = "▲" if delta_pct >= 0 else "▼"

with k1:
    st.markdown(f"""<div class='metric-card'><h3>Latest Load</h3>
    <p>{latest_val:,.0f} MW</p><span class='{delta_class}'>{arrow} {delta_pct:+.2f}% vs prev hr</span></div>""",
                unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class='metric-card'><h3>Selected Window Avg</h3>
    <p>{filtered['Power_Load'].mean():,.0f} MW</p></div>""" if not filtered.empty else
                "<div class='metric-card'><h3>Selected Window Avg</h3><p>—</p></div>", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class='metric-card'><h3>Selected Window Peak</h3>
    <p>{filtered['Power_Load'].max():,.0f} MW</p></div>""" if not filtered.empty else
                "<div class='metric-card'><h3>Selected Window Peak</h3><p>—</p></div>", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class='metric-card'><h3>Best Model (30d eval)</h3>
    <p>{best_model_row['Model']}</p></div>""", unsafe_allow_html=True)
with k5:
    st.markdown(f"""<div class='metric-card'><h3>Best Model R²</h3>
    <p>{best_model_row['R2']:.3f}</p></div>""", unsafe_allow_html=True)

st.write("")

# ------------------------------------------------------------------ TABS
tab_history, tab_compare, tab_forecast, tab_patterns, tab_data = st.tabs(
    ["📈 Live History", "🏆 Model Comparison", "🔮 Forecast", "🧭 Load Patterns", "📋 Raw Data"]
)

# ---- TAB 1: History
with tab_history:
    st.subheader(f"Power Load — aggregated by {granularity}")
    if agg_df.empty:
        st.warning("No data matches the current filters. Loosen a filter in the sidebar.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=agg_df["Timestamp"], y=agg_df["Power_Load"],
            mode="lines", name="Power Load",
            line=dict(color="#00e676", width=2),
            fill="tozeroy", fillcolor="rgba(0,230,118,0.08)",
        ))
        fig.update_layout(
            template="plotly_dark", height=460,
            margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="Timestamp", yaxis_title="Power Load (MW)",
            hovermode="x unified",
            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
        )
        st.plotly_chart(fig, width='stretch')

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Distribution of Power Load (filtered)**")
            fig_hist = px.histogram(filtered, x="Power_Load", nbins=50, color_discrete_sequence=["#58a6ff"])
            fig_hist.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=10, b=10),
                                    plot_bgcolor="#0d1117", paper_bgcolor="#0d1117")
            st.plotly_chart(fig_hist, width='stretch')
        with c2:
            st.markdown("**Weekly Boxplot — Load by Day of Week**")
            box_df = filtered.copy()
            box_df["DOW"] = box_df["Weekday_Number"].map(dict(enumerate(dow_labels)))
            fig_box = px.box(box_df, x="DOW", y="Power_Load", category_orders={"DOW": dow_labels},
                              color_discrete_sequence=["#f778ba"])
            fig_box.update_layout(template="plotly_dark", height=320, margin=dict(l=10, r=10, t=10, b=10),
                                   plot_bgcolor="#0d1117", paper_bgcolor="#0d1117")
            st.plotly_chart(fig_box, width='stretch')

# ---- TAB 2: Model Comparison
with tab_compare:
    st.subheader("5-Model Comparison — Common 30-Day Evaluation Window")
    st.caption(f"Evaluation window: {pd.to_datetime(meta['eval_window_start']).date()} → "
               f"{pd.to_datetime(meta['eval_window_end']).date()}")

    cc1, cc2 = st.columns([0.4, 0.6])
    with cc1:
        st.dataframe(
            comparison.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.4f}"})
            .background_gradient(subset=["RMSE"], cmap="RdYlGn_r"),
            width='stretch', hide_index=True
        )
        fig_bar = px.bar(comparison, x="Model", y="RMSE", color="Model",
                          color_discrete_map=MODEL_COLORS, text_auto=".1f")
        fig_bar.update_layout(template="plotly_dark", height=340, showlegend=False,
                               margin=dict(l=10, r=10, t=30, b=10),
                               plot_bgcolor="#0d1117", paper_bgcolor="#0d1117")
        st.plotly_chart(fig_bar, width='stretch')

    with cc2:
        st.markdown("**Actual vs. Predicted — 30-Day Hold-out**")
        models_to_show = st.multiselect(
            "Models to plot", list(MODEL_COLORS.keys())[1:], default=list(MODEL_COLORS.keys())[1:]
        )
        fig_eval = go.Figure()
        fig_eval.add_trace(go.Scatter(
            x=eval_preds["Timestamp"], y=eval_preds["Actual"], name="Actual",
            line=dict(color=MODEL_COLORS["Actual"], width=3)
        ))
        for m in models_to_show:
            if m in eval_preds.columns:
                fig_eval.add_trace(go.Scatter(
                    x=eval_preds["Timestamp"], y=eval_preds[m], name=m,
                    line=dict(color=MODEL_COLORS.get(m), width=1.6, dash="dot")
                ))
        fig_eval.update_layout(template="plotly_dark", height=460, hovermode="x unified",
                                margin=dict(l=10, r=10, t=30, b=10),
                                plot_bgcolor="#0d1117", paper_bgcolor="#0d1117")
        st.plotly_chart(fig_eval, width='stretch')

    st.info(f"🏆 **{best_model_row['Model']}** is the production-selected model — lowest RMSE "
            f"({best_model_row['RMSE']:.1f} MW) and highest R² ({best_model_row['R2']:.3f}) on the shared 30-day test window.")

# ---- TAB 3: Forecast
with tab_forecast:
    st.subheader(f"{horizon_choice} Forecast — {forecast_model}")

    if forecast_model == "XGBoost (Tuned)" and horizon_hours != 720:
        with st.spinner(f"Generating live {horizon_choice} recursive forecast..."):
            forecast_df = recursive_forecast(horizon_hours, (xgb_model, xgb_cols))
    else:
        full_forecast = load_forecast(forecast_model)
        forecast_df = full_forecast.head(horizon_hours)

    recent_actual = df.tail(24 * 14)[["Timestamp", "Power_Load"]]

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=recent_actual["Timestamp"], y=recent_actual["Power_Load"],
        name="Recent Actual (14d)", line=dict(color="#e6edf3", width=2)
    ))
    fig_fc.add_trace(go.Scatter(
        x=forecast_df["Timestamp"], y=forecast_df["Forecast_Power_Load"],
        name=f"Forecast ({forecast_model})", line=dict(color=MODEL_COLORS.get(forecast_model, "#00e676"), width=2.4, dash="dash")
    ))
    fig_fc.add_vline(x=df["Timestamp"].max(), line_dash="dot", line_color="#8b949e")
    fig_fc.update_layout(template="plotly_dark", height=460, hovermode="x unified",
                          margin=dict(l=10, r=10, t=30, b=10), yaxis_title="Power Load (MW)",
                          plot_bgcolor="#0d1117", paper_bgcolor="#0d1117")
    st.plotly_chart(fig_fc, width='stretch')

    fc1, fc2, fc3 = st.columns(3)
    fc1.metric("Forecast Avg", f"{forecast_df['Forecast_Power_Load'].mean():,.0f} MW")
    fc2.metric("Forecast Peak", f"{forecast_df['Forecast_Power_Load'].max():,.0f} MW")
    fc3.metric("Forecast Low", f"{forecast_df['Forecast_Power_Load'].min():,.0f} MW")

    with st.expander("View forecast table"):
        st.dataframe(forecast_df, width='stretch', hide_index=True)
    st.download_button("⬇️ Download forecast CSV", forecast_df.to_csv(index=False),
                        file_name=f"forecast_{forecast_model.replace(' ', '_')}_{horizon_choice.replace(' ', '')}.csv")

# ---- TAB 4: Load Patterns
with tab_patterns:
    st.subheader("Seasonality & Pattern Explorer")

    st.markdown("**Hourly Load Profile by Season** — how the daily shape changes across the year")
    season_order = ["Winter", "Spring", "Summer", "Autumn"]
    season_colors = {"Winter": "#58a6ff", "Spring": "#3fb950", "Summer": "#f778ba", "Autumn": "#ffa657"}
    season_hourly = (
        filtered.groupby(["Season_Group", "Hour_Slot"])["Power_Load"].mean().reset_index()
    )
    fig_season = go.Figure()
    for s in season_order:
        sub = season_hourly[season_hourly["Season_Group"] == s]
        if sub.empty:
            continue
        fig_season.add_trace(go.Scatter(
            x=sub["Hour_Slot"], y=sub["Power_Load"], mode="lines+markers", name=s,
            line=dict(color=season_colors[s], width=2.4)
        ))
    fig_season.update_layout(
        template="plotly_dark", height=380, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Hour of Day", yaxis_title="Avg Power Load (MW)", hovermode="x unified",
        plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
    )
    st.plotly_chart(fig_season, width='stretch')
    summer_peak = season_hourly[season_hourly["Season_Group"] == "Summer"]["Power_Load"].max()
    winter_peak = season_hourly[season_hourly["Season_Group"] == "Winter"]["Power_Load"].max()
    if pd.notna(summer_peak) and pd.notna(winter_peak):
        st.caption(f"Summer peak-hour avg: {summer_peak:,.0f} MW  •  Winter peak-hour avg: {winter_peak:,.0f} MW  •  "
                   f"Summer and Winter profiles shift and reshape relative to Spring/Autumn, driven by AC and heating load.")

    st.markdown("**Year-over-Year Long-Term Trend**")
    yearly = filtered.groupby("Calendar_Year")["Power_Load"].agg(["mean", "max", "min"]).reset_index()
    fig_year = go.Figure()
    fig_year.add_trace(go.Scatter(x=yearly["Calendar_Year"], y=yearly["max"], name="Yearly Peak",
                                   line=dict(color="#f85149", width=1.6, dash="dot")))
    fig_year.add_trace(go.Scatter(x=yearly["Calendar_Year"], y=yearly["mean"], name="Yearly Avg",
                                   line=dict(color="#00e676", width=2.4)))
    fig_year.add_trace(go.Scatter(x=yearly["Calendar_Year"], y=yearly["min"], name="Yearly Min",
                                   line=dict(color="#58a6ff", width=1.6, dash="dot")))
    fig_year.update_layout(template="plotly_dark", height=340, margin=dict(l=10, r=10, t=10, b=10),
                            xaxis_title="Year", yaxis_title="Power Load (MW)", hovermode="x unified",
                            plot_bgcolor="#0d1117", paper_bgcolor="#0d1117")
    st.plotly_chart(fig_year, width='stretch')

    p1, p2 = st.columns(2)
    with p1:
        st.markdown("**Average Load by Hour of Day**")
        hourly_avg = filtered.groupby("Hour_Slot")["Power_Load"].mean().reset_index()
        fig_h = px.line(hourly_avg, x="Hour_Slot", y="Power_Load", markers=True,
                         color_discrete_sequence=["#00e676"])
        fig_h.update_layout(template="plotly_dark", height=340, margin=dict(l=10, r=10, t=10, b=10),
                             plot_bgcolor="#0d1117", paper_bgcolor="#0d1117")
        st.plotly_chart(fig_h, width='stretch')

        st.markdown("**Average Load by Month**")
        month_avg = filtered.groupby("Month_Index")["Power_Load"].mean().reset_index()
        fig_m = px.bar(month_avg, x="Month_Index", y="Power_Load", color_discrete_sequence=["#58a6ff"])
        fig_m.update_layout(template="plotly_dark", height=340, margin=dict(l=10, r=10, t=10, b=10),
                             plot_bgcolor="#0d1117", paper_bgcolor="#0d1117")
        st.plotly_chart(fig_m, width='stretch')

    with p2:
        st.markdown("**Hour × Day-of-Week Heatmap**")
        heat = filtered.copy()
        heat["DOW"] = heat["Weekday_Number"].map(dict(enumerate(dow_labels)))
        pivot = heat.pivot_table(index="DOW", columns="Hour_Slot", values="Power_Load", aggfunc="mean")
        pivot = pivot.reindex(dow_labels)
        fig_heat = px.imshow(pivot, color_continuous_scale="Viridis", aspect="auto",
                              labels=dict(color="MW"))
        fig_heat.update_layout(template="plotly_dark", height=340, margin=dict(l=10, r=10, t=10, b=10),
                                plot_bgcolor="#0d1117", paper_bgcolor="#0d1117")
        st.plotly_chart(fig_heat, width='stretch')

        st.markdown("**Holiday vs Non-Holiday Load**")
        hol = filtered.groupby("Is_Holiday")["Power_Load"].mean().reset_index()
        hol["Is_Holiday"] = hol["Is_Holiday"].map({0: "Non-Holiday", 1: "Holiday"})
        fig_hol = px.bar(hol, x="Is_Holiday", y="Power_Load", color="Is_Holiday",
                          color_discrete_sequence=["#58a6ff", "#f778ba"])
        fig_hol.update_layout(template="plotly_dark", height=340, showlegend=False,
                               margin=dict(l=10, r=10, t=10, b=10),
                               plot_bgcolor="#0d1117", paper_bgcolor="#0d1117")
        st.plotly_chart(fig_hol, width='stretch')

# ---- TAB 5: Raw Data
with tab_data:
    st.subheader("Filtered Raw Data")
    st.caption(f"{len(filtered):,} rows match the current sidebar filters.")
    st.dataframe(
        filtered[["Timestamp", "Power_Load", "Hour_Slot", "Weekday_Number", "Season_Group",
                  "Is_Holiday", "Weekend_Flag"]].sort_values("Timestamp", ascending=False),
        width='stretch', height=460
    )
    st.download_button("⬇️ Download filtered data CSV", filtered.to_csv(index=False),
                        file_name="filtered_power_load.csv")

# ------------------------------------------------------------------ LIVE MODE AUTO-REFRESH
if live_mode:
    st.caption(f"🔄 Auto-refreshing every {refresh_secs}s to simulate a live meter feed...")
    time.sleep(refresh_secs)
    st.rerun()
