# Import relevant packages
import pandas as pd

def preprocess(df: pd.DataFrame) -> pd.DataFrame:

    # Convert date to datetime64 DataType
    df["date"] = pd.to_datetime(df["date"])

    # Set date as index
    df.set_index("date", inplace=True)

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
