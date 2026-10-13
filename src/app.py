import streamlit as st
from pathlib import Path
import joblib
import pandas as pd


@st.cache_resource
def load_model():
    '''Method for fetching the saved .pkl model'''

    # Get saved model path
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"

    # Fetch and return model
    model = joblib.load(MODEL_PATH)
    return model

@st.cache_resource
def load_data(name):
    '''Method for fetching the test and calibration data'''

    # Get saved csv file path
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # Fetch and return
    df = pd.read_csv(PROJECT_ROOT/"data"/'processed'/name)
    return df

# Load model using cache first
model = load_model()

# Load test and calibration set
test_set = load_data('test_set.csv')
cal_set = load_data('cal_set.csv')

st.title("⚡Live Energy Forecasting Demo")
st.write('Simulated real-time energy forecasting using unseen UCI Appliances Energy dataset observations.')


speed = st.slider("Select playback speed (seconds per 10-minutes)", 1, 5, 2)