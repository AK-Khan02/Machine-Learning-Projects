from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

class RegressionMetrics:
    def __init__(self, y_true, y_pred):
        """
        Initialize the metrics class with true values and predictions.

        Parameters:
        - y_true: True values.
        - y_pred: Predicted values.
        """
        self.y_true = y_true
        self.y_pred = y_pred

    def mse(self):
        return mean_squared_error(self.y_true, self.y_pred)

    def rmse(self):
        return np.sqrt(self.mse())

    def mae(self):
        return mean_absolute_error(self.y_true, self.y_pred)

    def get_all_metrics(self):
        """
        Returns all metrics in a dictionary.
        """
        return {
            'MSE': self.mse(),
            'RMSE': self.rmse(),
            'MAE': self.mae()
        }
