# Import relevant packages
import pandas as pd
import numpy as np

def preprocess_energy(df: pd.DataFrame) -> pd.DataFrame:

    '''
    Dataset Name: UCI Appliances Energy Prediction
    Dataset Source: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction
    ---

    This method does the following preprocessing steps:
    1. Convert 'date' column from string DataType to datetime64
    2. Set 'date' as index
    3. Extract calendar feature (hour, month, day)
    4. Drop columns representing Chievres Airport readings
    5. Aggregate temperature and humidity features into similar groups
    6. Drop unaggregated features

    Justifications and EDA are documented in:
    energy-forecasting/notebooks/01_data_preprocessing.ipynb
    '''

    # Convert date to datetime64 DataType
    df["date"] = pd.to_datetime(df["date"])

    # Set date as index
    df.set_index("date", inplace=True)

    # Extract calendar feature
    df['hour'] = df.index.hour
    df['month'] = df.index.month_name()
    df['day'] = df.index.day_name()

    # Define features representing Chievres Airport
    chievres = ['T_out','Press_mm_hg','RH_out','Windspeed','Visibility','Tdewpoint']

    # Drop columns
    df = df.drop(chievres, axis = 1)

    # Define grouping for aggregation
    groups = {
    "indoor": [1, 3, 4, 7, 8, 9],
    "liv_room": [2],
    "bathroom": [5],
    "outdoor": [6],
    }

    # Perform aggregation using mean
    for name, indices in groups.items():
        # Mean Temperature
        temp_cols = [f"T{i}" for i in indices]
        df[f"T_{name}"] = df[temp_cols].mean(axis=1)

        # Mean Humidity
        rh_cols = [f"RH_{i}" for i in indices]
        df[f"RH_{name}"] = df[rh_cols].mean(axis=1)

    # Drop old, unaggregated features
    for i in range (9):
        df = df.drop([f'T{i+1}',  f'RH_{i+1}'], axis = 1)

    # Confirmation message
    print('Preprocessing completed.')

    # Return preprocessed dataset
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    '''
        Dataset Name: UCI Appliances Energy Prediction
        Dataset Source: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction
        ---
    
        This method does the following feature engineering steps:
        1. Keep 'hour' as a predictive feature for the model, but discard the 'day'
        2. Downsample to 1 sample/30 min, aggregated by max
        3. Create lagged variable for 30-120 minutes (after downsampling), as well as 1, 5 and 7 days
        4. Keep light lagged variables short-term (30 - 60 minutes back)
        5. Keep Only 90-120 minutes Lagging for Bathroom Humidity
    
        Justifications and EDA are documented in:
        energy-forecasting/notebooks/02_eda.ipynb
    '''

    # Discard 'day'
    df = df.drop('day', axis = 1)

    # Downsample to 1 sample/30 min, aggregated by max
    agg = {}

    # If numeric, take max. Else, take the first row values
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number):
            agg[col] = "max"
        else:
            agg[col] = "last"   

    df = df.resample("30min").agg(agg)

    # # Create lagged variable: 30-120 minutes, and 1, 5, 7 days
    # lag = [1, 2, 3, 4, 48, 240, 336]
    # lag_lab = ['30min', '1h', '1h30min', '2h', '1d', '5d', '7d']

    # for col in df.select_dtypes('number').columns:
    #     for step, label in lag, lag_lab:
    #         df[f'{col}_{label}'] = 