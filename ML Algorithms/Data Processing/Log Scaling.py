import numpy as np

def log_scale(X, small_value=1e-6):
    """
    Apply log scaling to the data. A small value is added to avoid log(0).

    Parameters:
    - X : numpy array of shape (n_samples, n_features).
    - small_value : A small constant to add to the data before taking the log to avoid log(0).

    Returns:
    - Log-scaled data.
    """
    return np.log(X + small_value)

# Sample data with outliers
data_with_outliers = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [100, 200, 300]], dtype=float)

# Log Scaling
# Note: Log scaling expects all values to be positive.
data_log_scaled = log_scale(data_with_outliers)
print("\nData after Log Scaling:\n", data_log_scaled)
