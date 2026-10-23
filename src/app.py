import streamlit as st
from pathlib import Path
import joblib
import pandas as pd
import sys
import time
import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title="Live Energy Forecast", page_icon="⚡", layout="wide")

# Add the root directory to Python's path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

# Our wrapper methods
from src.preprocess import preprocess_energy, feature_engineering, xgb_preprocess
from interval_prediction import get_conformal_prediction

# Default value if no saved index session state is present
if "current_index" not in st.session_state:
    st.session_state.current_index = 0

if "actual_history" not in st.session_state:
    st.session_state.actual_history = {}

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = {}

if "lower_history" not in st.session_state:
    st.session_state.lower_history = {}

if "upper_history" not in st.session_state:
    st.session_state.upper_history = {}

# Define room
ROOMS = {
    1: "Kitchen", 2: "Living room", 3: "Laundry room",
    4: "Office room", 5: "Bathroom", 6: "Outside",
    7: "Ironing room", 8: "Teenager room", 9: "Parents room",
}

@st.cache_resource
def load_model():
    '''This is the model and encoder loader'''
    MODEL_PATH = root_dir / "models" / "xgb_best.pkl"
    ENCODER_PATH = root_dir / "models" / "month_encoder.pkl"
    return joblib.load(MODEL_PATH), joblib.load(ENCODER_PATH)

def load_data(name):
    '''This is the dataset loader'''
    return pd.read_csv(root_dir / "data" / "processed" / name)

def format_date(value):
    '''This method format the date in readable format'''
    return pd.to_datetime(value).strftime("%A, %d %B %Y  ·  %I:%M %p")

def show_sensor_readings(row):
    '''Show sensor readings using native Streamlit metrics in a tight grid'''
    # Group rooms into tabs for a cleaner UI
    tab1, tab2, tab3 = st.tabs(["Common Areas", "Private Rooms", "Utility & Outside"])

    with tab1:
        cols = st.columns(3)
        for col, n in zip(cols, [1, 2, 5]): # Kitchen, Living, Bath
            col.metric(f"{ROOMS[n]} Temp", f"{row[f'T{n}']:.1f} °C")
            col.metric(f"{ROOMS[n]} Humidity", f"{row[f'RH_{n}']:.1f} %")

    with tab2:
        cols = st.columns(3)
        for col, n in zip(cols, [4, 8, 9]): # Office, Teen, Parents
            col.metric(f"{ROOMS[n]} Temp", f"{row[f'T{n}']:.1f} °C")
            col.metric(f"{ROOMS[n]} Humidity", f"{row[f'RH_{n}']:.1f} %")

    with tab3:
        cols = st.columns(3)
        for col, n in zip(cols, [3, 7, 6]): # Laundry, Ironing, Outside
            col.metric(f"{ROOMS[n]} Temp", f"{row[f'T{n}']:.1f} °C")
            col.metric(f"{ROOMS[n]} Humidity", f"{row[f'RH_{n}']:.1f} %")

def show_prediction(lower_bounds, upper_bounds, horizon):
    '''Show the forecast using clean metric blocks with conformal intervals'''
    # Extract the final prediction matching the target horizon
    final_lower = lower_bounds[-1]
    final_upper = upper_bounds[-1]

    # Display as a single metric
    st.metric(f"In {horizon} min", f"{final_lower:.2f}Wh - {final_upper:.2f}Wh")

def get_time_series_chart():
    '''Returns the Plotly figure for actual and predicted energy consumption'''
    actual_history = st.session_state.actual_history
    prediction_history = st.session_state.prediction_history
    lower_history = st.session_state.lower_history
    upper_history = st.session_state.upper_history

    if not actual_history and not prediction_history:
        return None

    actual_series = pd.Series(actual_history, dtype="float64")
    prediction_series = pd.Series(prediction_history, dtype="float64")
    lower_series = pd.Series(lower_history, dtype="float64")
    upper_series = pd.Series(upper_history, dtype="float64")

    actual_series.index = pd.to_datetime(actual_series.index)
    prediction_series.index = pd.to_datetime(prediction_series.index)
    lower_series.index = pd.to_datetime(lower_series.index)
    upper_series.index = pd.to_datetime(upper_series.index)

    chart_df = pd.concat(
        [
            actual_series.rename("Actual"),
            prediction_series.rename("Predicted"),
            lower_series.rename("Lower Bound"),
            upper_series.rename("Upper Bound"),
        ],
        axis=1,
    ).sort_index()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["Actual"],
            mode="lines",
            name="Actual",
            connectgaps=False,
            hovertemplate="Time: %{x}<br>Actual: %{y:.2f} Wh<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["Predicted"],
            mode="lines",
            name="Predicted",
            connectgaps=False,
            hovertemplate="Time: %{x}<br>Predicted: %{y:.2f} Wh<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["Lower Bound"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            connectgaps=False,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=chart_df.index,
            y=chart_df["Upper Bound"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(31, 119, 180, 0.20)",
            name="Conformal Prediction",
            connectgaps=False,
            hovertemplate="Time: %{x}<br>Upper Bound: %{y:.2f} Wh<extra></extra>",
        )
    )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Energy Consumption (Wh)",
        height=450,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=20, r=20, t=50, b=20),
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    return fig

# Load data
model, encoder = load_model()
test_set = load_data("test_set.csv")
cal_set = load_data("cal_set.csv")

# Preprocess calibration set for Conformal Prediction
cal_processed_df = preprocess_energy(cal_set.copy())
cal_processed_df = feature_engineering(cal_processed_df)
X_cal = xgb_preprocess(cal_processed_df, encoder)
y_cal = cal_processed_df['Appliances']

# Set title
st.title("⚡ Live Energy Forecasting Dashboard")

# Render static title outside the placeholder to prevent layout flickering
st.markdown("### 📊 Actual vs Predicted Energy Consumption")
chart_placeholder = st.empty()

initial_fig = get_time_series_chart()
if initial_fig is not None:
    chart_placeholder.plotly_chart(initial_fig, use_container_width=True)
else:
    chart_placeholder.info("The time-series graph will appear when the simulation starts.")

# Slider to select playback speed and forecast horizon
st.markdown("### ⚙️ Simulation Settings")
col_speed, col_horizon = st.columns(2)

with col_speed:
    speed = st.select_slider(
        "Playback speed (seconds per 10 mins)",
        options=[0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5],
        value=0.05,
        format_func=lambda s: f"{s}s",
    )

with col_horizon:
    horizon = st.slider(
        "Forecast horizon (minutes)",
        min_value=30,
        max_value=120,
        step=30,
    )

st.divider()

# Top level metrics container
top_col1, top_col2 = st.columns([2, 1])
today_placeholder = top_col1.empty()
lights_placeholder = top_col2.empty()

st.markdown("<br>", unsafe_allow_html=True)

# Main dashboard layout (Left side: Predictions, Right side: Sensors)
main_col1, main_col2 = st.columns([1.2, 1])

# Create static headers outside placeholders
with main_col1:
    st.markdown("### 📈 Energy Consumption Forecast")
    st.markdown("#### Forecast Breakdown")
    prediction_placeholder = st.empty()

with main_col2:
    st.markdown("### 🏠 Live Home Sensor Readings")
    row_placeholder = st.empty()

# Simulation loop
history_raw = cal_set.copy()

if st.session_state.current_index > 0:
    history_raw = pd.concat(
        [history_raw, test_set.iloc[:st.session_state.current_index]]
    )

for i in range(st.session_state.current_index, len(test_set)):

    # Calculate period (steps)
    period = int(horizon / 30)

    # Stop running if simulation ended (ran out of test row)
    available_periods = min(period, len(test_set) - i - 1)
    if available_periods <= 0:
        st.session_state.current_index = len(test_set)
        break

    # Append current reading to history
    history_raw = pd.concat([history_raw, test_set.iloc[[i]]])
    current_row = history_raw.iloc[-1]

    current_date = pd.to_datetime(current_row["date"])
    current_date_key = current_date.isoformat()

    st.session_state.actual_history[current_date_key] = float(
        current_row["Appliances"]
    )

    # Update top metrics directly without clearing a container block
    today_placeholder.subheader(f"🕒 {format_date(current_row['date'])}")
    lights_placeholder.metric(
        "💡 Current Lights Usage",
        f"{current_row['lights']:.0f} Wh",
    )

    # Update Sensors (Right Column)
    with row_placeholder.container():
        show_sensor_readings(current_row)

    # Real-time preprocessing
    sensor_forecast = pd.concat(
        [history_raw, test_set.iloc[i + 1:i + 1 + available_periods]]
    )
    sensor_forecast = preprocess_energy(sensor_forecast)
    sensor_forecast = feature_engineering(sensor_forecast)
    sensor_forecast = xgb_preprocess(sensor_forecast, encoder)

    # Get sensor forecast from API
    api_processed = sensor_forecast.tail(available_periods)

    # Make prediction using the conformal prediction wrapper
    y_pred, lower_bounds, upper_bounds = get_conformal_prediction(
        model,
        X_cal,
        y_cal,
        api_processed,
    )

    y_pred = pd.Series(y_pred).to_numpy().flatten()
    lower_bounds = pd.Series(lower_bounds).to_numpy().flatten()
    upper_bounds = pd.Series(upper_bounds).to_numpy().flatten()

    forecast_dates = pd.to_datetime(
        test_set.iloc[i + 1:i + 1 + available_periods]["date"]
    ).tolist()

    if not (
        len(y_pred) == len(lower_bounds) == len(upper_bounds) == len(forecast_dates)
    ):
        raise ValueError(
            "The number of predictions, prediction bounds and forecast timestamps do not match."
        )

    for j, forecast_date in enumerate(forecast_dates):
        forecast_date_key = pd.Timestamp(forecast_date).isoformat()

        st.session_state.prediction_history[forecast_date_key] = float(y_pred[j])
        st.session_state.lower_history[forecast_date_key] = float(lower_bounds[j])
        st.session_state.upper_history[forecast_date_key] = float(upper_bounds[j])

    # Fetch updated figure and overwrite chart_placeholder cleanly
    updated_fig = get_time_series_chart()
    if updated_fig is not None:
        chart_placeholder.plotly_chart(updated_fig, use_container_width=True)

    # Update Predictions (Left Column)
    with prediction_placeholder.container():
        show_prediction(
            lower_bounds,
            upper_bounds,
            available_periods * 30,
        )

    st.session_state.current_index = i + 1

    time.sleep(speed)

if st.session_state.current_index >= len(test_set):
    st.session_state.current_index = len(test_set)