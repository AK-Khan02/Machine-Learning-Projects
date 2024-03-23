import numpy as np
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import mode

class BaggingEnsemble:
    def __init__(self, base_estimator=None, n_estimators=10, random_state=None):
        self.base_estimator = base_estimator if base_estimator else DecisionTreeClassifier()
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models = []
        self.random_states = np.random.randint(0, 10000, size=n_estimators) if random_state is None else np.full(n_estimators, random_state)

    def fit(self, X, y):
        np.random.seed(self.random_state)
        self.models = []
        for i in range(self.n_estimators):
            # Bootstrap sample
            indices = np.random.randint(0, len(X), len(X))
            X_sample, y_sample = X[indices], y[indices]

            # Clone the base estimator and fit on the sample
            model = clone(self.base_estimator)
            model.fit(X_sample, y_sample)
            self.models.append(model)

    def predict(self, X):
        # Collect predictions from each model
        predictions = np.array([model.predict(X) for model in self.models])
        
        # Majority vote or average
        if isinstance(self.base_estimator, DecisionTreeClassifier):
            return mode(predictions, axis=0)[0][0]
        else:  # For regression models
            return np.mean(predictions, axis=0)
