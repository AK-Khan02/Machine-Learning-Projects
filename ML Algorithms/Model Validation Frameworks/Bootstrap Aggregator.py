import numpy as np
from sklearn.utils import resample
from sklearn.base import clone
from sklearn.metrics import accuracy_score

class BootstrapAggregator:
    def __init__(self, base_estimator, n_estimators=10, random_state=None):
        """
        Initialize the Bootstrap Aggregator (Bagging) class.

        Parameters:
        - base_estimator: The machine learning model to be used for training on bootstrap samples.
        - n_estimators: Number of bootstrap samples and hence the number of models to train.
        - random_state: Random state for reproducibility of bootstrap samples.
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models = []
        self.random_states = np.random.randint(0, 10000, size=n_estimators) if random_state is None else np.full(n_estimators, random_state)

    def fit(self, X, y):
        """
        Fit the Bagging ensemble of models.

        Parameters:
        - X: Feature matrix of shape (n_samples, n_features).
        - y: Target vector of shape (n_samples,).
        """
        self.models = []
        for i in range(self.n_estimators):
            # Create bootstrap sample
            X_sample, y_sample = resample(X, y, replace=True, n_samples=len(X), random_state=self.random_states[i])
            
            # Clone the base estimator to ensure independence between the models
            model = clone(self.base_estimator)
            model.fit(X_sample, y_sample)
            self.models.append(model)

    def predict(self, X):
        """
        Predict the target for X by aggregating predictions from the ensemble of models.

        Parameters:
        - X: Feature matrix of shape (n_samples, n_features).

        Returns:
        - Aggregated predictions.
        """
        # Collect predictions from each model
        predictions = np.array([model.predict(X) for model in self.models])
        
        # Aggregate predictions (majority voting for classification)
        agg_predictions = np.squeeze(np.apply_along_axis(lambda x: np.bincount(x, minlength=2).argmax(), arr=predictions, axis=0))

        return agg_predictions

    def score(self, X, y):
        """
        Evaluate the accuracy of the aggregated model.

        Parameters:
        - X: Feature matrix of shape (n_samples, n_features).
        - y: True target values.

        Returns:
        - Accuracy of the model on the given dataset.
        """
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

# Example usage:
"""
from sklearn.tree import DecisionTreeClassifier

# Base estimator
base_estimator = DecisionTreeClassifier()

# Bootstrap Aggregator
bagging = BootstrapAggregator(base_estimator=base_estimator, n_estimators=100, random_state=42)

# Fit the model
bagging.fit(X_train, y_train)

# Evaluate the model
accuracy = bagging.score(X_test, y_test)
print(f'Bagging Model Accuracy: {accuracy}')
"""
