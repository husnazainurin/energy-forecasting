import streamlit as st
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sys
import time

# Add the root directory (one level up from notebooks/) to Python's path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.preprocess import preprocess_energy, feature_engineering, arimax_preprocess

if "current_index" not in st.session_state:
    st.session_state.current_index = 0

# Friendly names for the UCI Appliances Energy sensors
ROOMS = {
    1: "Kitchen",
    2: "Living room",
    3: "Laundry room",
    4: "Office room",
    5: "Bathroom",
    6: "Outside (north)",
    7: "Ironing room",
    8: "Teenager room",
    9: "Parents room",
}

@st.cache_resource
def load_model():
    '''Method for fetching the saved .pkl model'''
    MODEL_PATH = root_dir / "models" / "model.pkl"
    ENCODER_PATH = root_dir / "models" / "encoder.pkl"

    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    return model, encoder

def load_data(name):
    '''Method for fetching the test and calibration data'''
    df = pd.read_csv(root_dir / "data" / "processed" / name)
    return df

def format_date(value):
    '''Turn 2016-04-02 22:50:00 into Saturday, 02 April 2016 · 10:50 PM'''
    return pd.to_datetime(value).strftime("%A, %d %B %Y  ·  %I:%M %p")

def show_sensor_readings(row):
    '''Show the current sensor readings in a simple, readable layout'''
    st.subheader("Current Sensor Readings")

    st.metric("💡 Lights energy use", f"{row['lights']:.0f} Wh")

    for start in (1, 4, 7):
        cols = st.columns(3)
        for col, n in zip(cols, range(start, start + 3)):
            with col:
                st.markdown(f"**{ROOMS[n]}**")
                st.write(f"🌡️ {row[f'T{n}']:.1f} °C")
                st.write(f"💧 {row[f'RH_{n}']:.1f} %")

def show_prediction(prediction, horizon):
    '''Show the forecast in plain sentences, one line per 30-minute step'''

    values = np.asarray(prediction).ravel()
    for step, value in enumerate(values, start=1):
        minutes = step * 30
        st.subheader(f"**Consumption prediction in {minutes} minutes:** {value:.1f} Wh")


# Load model using cache first
model, encoder = load_model()

# Load test and calibration set
test_set = load_data("test_set.csv")
cal_set = load_data("cal_set.csv")

st.title("⚡ Live Energy Forecasting Demo")
st.write("Simulated real-time energy forecasting using unseen UCI Appliances Energy dataset observations.")

# Settings
col_speed, col_horizon = st.columns(2)
with col_speed:
    speed = st.select_slider(
        "Playback speed (seconds per 10 minutes)",
        options=[0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5],
        value=2,
        format_func=lambda s: f"{s}s",
        help="Lower = faster. Very low values may be limited by how fast the model can run.",
    )
with col_horizon:
    horizon = st.slider("Forecast horizon (minutes)", min_value=30, max_value=120, step=30)

st.divider()

# Live placeholders
prediction_placeholder = st.empty()
today_placeholder = st.empty()
row_placeholder = st.empty()


# FIX: Correctly reconstruct history_raw on Streamlit rerun by including previous iterations
history_raw = cal_set.copy()
if st.session_state.current_index > 0:
    history_raw = pd.concat([history_raw, test_set.iloc[:st.session_state.current_index]])

for i in range(st.session_state.current_index, len(test_set)):
    
    st.session_state.current_index = i

    # FIX: Calculate available periods to prevent out-of-bounds leakage at the end of the dataset
    period = int(horizon / 30)
    available_periods = min(period, len(test_set) - i - 1)
    
    # Break cleanly if there isn't enough future data left to forecast
    if available_periods <= 0:
        break

    # Append current row to history
    history_raw = pd.concat([history_raw, test_set.iloc[[i]]])

    # Print current time
    today_placeholder.markdown(f"Current Time🕒: {format_date(history_raw['date'].iloc[-1])}")

    # Display current sensor reading
    with row_placeholder.container():
        show_sensor_readings(history_raw.iloc[-1])

    # Get forecast of sensor reading across the allowed available horizon
    sensor_forecast = pd.concat([history_raw, test_set.iloc[i + 1:i + 1 + available_periods]])

    # FIX: Removed the redundant 'history' preprocessing since only 'sensor_forecast' 
    # needs to be preprocessed to supply features + lag to the ARIMAX model.
    sensor_forecast = preprocess_energy(sensor_forecast)
    sensor_forecast = feature_engineering(sensor_forecast)
    sensor_forecast, irr = arimax_preprocess(sensor_forecast, encoder)

    # Slice strictly the future available periods
    api_processed = sensor_forecast.tail(available_periods)

    # Make prediction across the dynamically verified horizon
    prediction = model.predict(n_periods=available_periods, X=api_processed)

    # Display point prediction across horizon
    with prediction_placeholder.container():
        show_prediction(prediction, available_periods * 30)

    time.sleep(speed)

# Reset when simulation fully completes
st.session_state.current_index = 0