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
    ENCODER_PATH = root_dir / "models" / "encoder.pkl"

    # Fetch and return model
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH)

    return model, encoder

def load_data(name):
    '''Method for fetching the test and calibration data'''

    # Fetch and return
    df = pd.read_csv(root_dir/"data"/'processed'/name)
    return df

if "current_index" not in st.session_state:
  st.session_state.current_index = 0


# Load model using cache first
model, encoder = load_model()

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

# Create history DataFrame and append calibration data
history_raw = cal_set.copy()
sensor_forecast = cal_set.copy()

for i in range (st.session_state.current_index, len(test_set)):

    # Save current iteration in-case loop interrupted
    st.session_state.current_index = i

    # Calculate forecast horizon step
    period = int(horizon/30)

    # Append current row to history
    history_raw = pd.concat([history_raw, test_set.iloc[[i]]])
    
    # Get forecast of sensor reading across horizon (this should come from external API)
    sensor_forecast = pd.concat([history_raw,test_set.iloc[i+1:i+1+period]])

    # Display current sensor reading
    label_sensor_placeholder.write('Current Sensor Readings: ')
    row_placeholder.dataframe(history_raw.iloc[-1])

    # Preprocess both history and future sensor reading
    history = preprocess_energy(history_raw)
    history = feature_engineering(history)

    sensor_forecast = preprocess_energy(sensor_forecast)
    sensor_forecast = feature_engineering(sensor_forecast)
    sensor_forecast, irr = arimax_preprocess(sensor_forecast, encoder)

    api_processed = sensor_forecast.tail(period)

    # Make prediction across horizon
    prediction = model.predict(n_periods=period, X=api_processed)

    # Display point prediction across horizon
    label_prediction_placeholder.write('Prediction:')
    
    prediction_placeholder.dataframe(prediction)

    time.sleep(speed)
    
    row_placeholder.empty()
    prediction_placeholder.empty()
    label_prediction_placeholder.empty()
    label_sensor_placeholder.empty()

st.session_state.current_index = 0

