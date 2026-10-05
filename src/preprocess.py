# Import relevant packages
import pandas as pd

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