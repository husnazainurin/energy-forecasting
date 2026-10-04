# Import relevant packages
import pandas as pd

def preprocess(df: pd.DataFrame) -> pd.DataFrame:

    # Convert date to datetime64 DataType
    df["date"] = pd.to_datetime(df["date"])

    # Set date as index
    df.set_index("date", inplace=True)

    # Define features representing Chievres Airport
    chievres = ['T_out','Press_mm_hg','RH_out','Windspeed','Visibility','Tdewpoint']
