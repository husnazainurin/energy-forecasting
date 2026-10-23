import numpy as np
from mapie.regression import SplitConformalRegressor

def get_conformal_prediction(model, X_cal, y_cal, X_test, confidence_level=0.95):
    """
    Wrapper method to generate conformal prediction intervals safely.
    """
    # Initialize the MAPIE regressor
    mapie_regressor = SplitConformalRegressor(
        estimator=model,
        confidence_level=confidence_level,
        prefit=True
    )
    
    # Conformalize using the calibration set
    mapie_regressor.conformalize(X_cal, y_cal)
    
    # Generate predictions and intervals for the test set
    y_pred, y_intervals = mapie_regressor.predict_interval(X_test)
    
    # y_intervals shape is (n_samples, 2, n_alpha)
    # Extract lower [0] and upper [1] bounds for the 1st alpha level [0]
    lower_bounds = np.maximum(0, y_intervals[:, 0, 0])
    upper_bounds = y_intervals[:, 1, 0]
    
    return y_pred, lower_bounds, upper_bounds