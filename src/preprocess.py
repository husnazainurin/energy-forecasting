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


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    '''
        Dataset Name: UCI Appliances Energy Prediction
        Dataset Source: https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction
        ---
    
        This method does the following feature engineering steps:
        1. Keep 'hour' as a predictive feature for the model, but discard the 'day'
        2. Downsample to 1 sample/30 min, aggregated by max
        3. Create lagged appliances variable for 30-120 minutes (after downsampling), as well as 1, 5 and 7 days
        4. Create short-term light lagged variables (30 - 60 minutes back)
        5. Create 90-120 minutes Lagging for Bathroom Humidity
        6. Drop resulting missing values caused by the shifting
    
        Justifications and EDA are documented in:
        energy-forecasting/notebooks/02_eda.ipynb
    '''

    # Discard 'day'
    df = df.drop(columns = ['day'])

    # Downsample to 1 sample/30 min, aggregated by max
    agg = {}
    numeric = df.select_dtypes('number').columns

    # If numeric, take max. Else, take the first row values
    for col in df.columns:
        if col in numeric:
            agg[col] = "max"
        else:
            agg[col] = "last"   

    df = df.resample("30min").agg(agg)

    # Create lagged variables: 30 min–2 h, and 1, 5, 7 days
    lag = [1, 2, 3, 4, 48, 240, 336]
    lag_lab = ['30min', '1h', '1h30min', '2h', '1d', '5d', '7d']

    for step, label in zip(lag, lag_lab):
        df[f'Appliances_{label}'] = df['Appliances'].shift(step)

        # Create 30 - 60 minutes back light lagged variables 
    lag = [1, 2]
    lag_lab = ['30min', '1h']

    for step, label in zip(lag, lag_lab):
        df[f'lights_{label}'] = df['lights'].shift(step)

    # Create 90-120 minutes lag for Bathroom Humidity
    lag = [3, 4]
    lag_lab = ['1h30min', '2h']

    for step, label in zip(lag, lag_lab):
        df[f'RH_bathroom_{label}'] = df['RH_bathroom'].shift(step)

    # Drop resulting missing values caused by shifting
    df = df.dropna()

    return df
