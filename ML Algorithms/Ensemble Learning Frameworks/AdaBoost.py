from sklearn.tree import DecisionTreeClassifier

class AdaBoost:
    def __init__(self, base_estimator=None, n_estimators=50, learning_rate=1.0):
        self.base_estimator = base_estimator if base_estimator else DecisionTreeClassifier(max_depth=1)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []
        self.model_weights = []

    def fit(self, X, y):
        sample_weights = np.full(len(y), 1 / len(y))
        
        for _ in range(self.n_estimators):
            model = clone(self.base_estimator)
            model.fit(X, y, sample_weight=sample_weights)
            pred = model.predict(X)

            # Compute weighted error rate
            incorrect = (pred != y)
            weighted_error = np.dot(sample_weights, incorrect) / sum(sample_weights)

            # Compute model weight
            model_weight = self.learning_rate * np.log((1 - weighted_error) / max(weighted_error, 1e-10))
            self.models.append(model)
            self.model_weights.append(model_weight)

            # Update sample weights
            sample_weights *= np.exp(model_weight * incorrect)
            sample_weights /= np.sum(sample_weights)  # Normalize weights

    def predict(self, X):
        model_preds = np.array([model.predict(X) for model in self.models])
        final_pred = np.dot(self.model_weights, model_preds)
        return np.sign(final_pred)
