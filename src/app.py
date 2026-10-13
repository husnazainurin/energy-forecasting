import streamlit as st
from pathlib import Path
import joblib
import pandas as pd
import sys
import time

# Add the root directory (one level up from notebooks/) to Python's path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.preprocess import preprocess_energy, feature_engineering, arimax_preprocess

if 'current_index' not in st.session_state:
    st.session_state.current_index = 0


@st.cache_resource
def load_model():
    '''Method for fetching the saved .pkl model'''

    # Get saved model path
    MODEL_PATH = root_dir / "models" / "model.pkl"

    # Fetch and return model
    model = joblib.load(MODEL_PATH)
    return model

@st.cache_resource
def load_data(name):
    '''Method for fetching the test and calibration data'''

    # Fetch and return
    df = pd.read_csv(root_dir/"data"/'processed'/name)
    df.set_index('date')
    return df

if "current_index" not in st.session_state:
  st.session_state.current_index = 0


# Load model using cache first
model = load_model()

# Load test and calibration set
test_set = load_data('test_set.csv')
cal_set = load_data('cal_set.csv')

st.title("⚡Live Energy Forecasting Demo")
st.write('Simulated real-time energy forecasting using unseen UCI Appliances Energy dataset observations.')


speed = st.slider("Select playback speed (seconds per 10-minutes)", 1, 5, 2)
horizon = st.slider("Select forecast horizon (minutes): ", min_value=30, max_value=120, step = 30)


# Live sensor readings placeholder
label_sensor_placeholder = st.empty()
row_placeholder = st.empty()
label_prediction_placeholder = st.empty()
prediction_placeholder = st.empty()

# Create history DataFrame
history = pd.DataFrame()

for i in range (st.session_state.current_index, len(test_set)):

    # Save current iteration in-case loop interrupted
    st.session_state.current_index = i

    # Update history to include current row
    row = test_set.iloc[[i]]
    history = pd.concat([history, row])
    history = history.set_index('date')
    st.session_state.current_index = i

    # Display current sensor reading
    label_sensor_placeholder.write('Current Sensor Readings: ')
    row_placeholder.dataframe(row)

    # Get forecast of next 30 minutes sensor reading (this should come from external API)
    period = int(horizon/30)
    sensor_forecast = test_set.iloc[(i+1) : (i + 1 + period)]
    sensor_forecast = sensor_forecast.set_index('date')
    sensor_forecast, irr = arimax_preprocess(sensor_forecast)


    # Preprocess historical data with current reading
    X, y = arimax_preprocess(history)


    # Make prediction across horizon
    prediction = model.predict(n_periods=period, X=sensor_forecast)

    # Display point prediction across horizon
    label_prediction_placeholder.write('Prediction:')
    prediction_placeholder.dataframe(prediction)

    time.sleep(speed)
    row_placeholder.empty()
    prediction_placeholder.empty()
    label_prediction_placeholder.empty()
    label_sensor_placeholder.empty()

st.session_state.current_index = 0

