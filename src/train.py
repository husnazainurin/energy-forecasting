import pandas as pd
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima
import numpy as np
import joblib
from sklearn.pipeline import Pipeline

# Add the root directory (one level up from notebooks/) to Python's path
root_dir = Path.cwd()
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from src.preprocess import preprocess_energy, feature_engineering, arimax_preprocess


# Import csv file
energy = pd.read_csv(f'{root_dir}/data/raw/energydata_complete.csv')

# Data preprocessing
energy = preprocess_energy(energy)

# Form feature through feature engineering
energy = feature_engineering(energy)

le = LabelEncoder()
le.fit(energy["month"])
X, y = arimax_preprocess(energy, le)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=False
)

# Start best hyperparameter search
arimax_best = auto_arima(
    y=y_train,
    X=X_train,
    d=0,          
    start_p=0, max_p=3,
    start_q=0, max_q=3,
    m=0, 
    seasonal=False,
    stepwise=True,  
    trace=True,     
    error_action="ignore",
    suppress_warnings=True
)

y_pred_arimax = arimax_best.predict(
    n_periods=len(y_test), 
    X=X_test
)

rmse = np.sqrt(mean_squared_error(y_test, y_pred_arimax))

print(f"Test RMSE: {rmse:.4f}")

joblib.dump(arimax_best, f"{root_dir}/models/model.pkl")
joblib.dump(le, f"{root_dir}/models/encoder.pkl")