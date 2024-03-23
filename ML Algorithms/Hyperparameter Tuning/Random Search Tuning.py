import numpy as np
from sklearn.model_selection import KFold
from scipy.stats import randint, uniform
from joblib import Parallel, delayed

def cross_val_score(model, X, y, scoring, cv):
    kf = KFold(n_splits=cv)
    scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        scores.append(scoring(model, X_test, y_test))

    return np.mean(scores)

def evaluate_model(model, param_dist, X, y, scoring, cv):
    # Randomly sample parameters
    params = {k: v.rvs() if hasattr(v, 'rvs') else v for k, v in param_dist.items()}
    model.set_params(**params)

    score = cross_val_score(model, X, y, scoring, cv)
    return score, params

def random_search(model, param_dist, X, y, scoring, cv=5, n_iter=100, n_jobs=-1):
    best_score = float('-inf') if scoring.higher_is_better else float('inf')
    best_params = None

    # Parallel execution of model evaluation
    results = Parallel(n_jobs=n_jobs)(delayed(evaluate_model)(model, param_dist, X, y, scoring, cv) for _ in range(n_iter))

    for score, params in results:
        if scoring.is_better(score, best_score):
            best_score = score
            best_params = params

    best_model = clone(model).set_params(**best_params)
    best_model.fit(X, y)  # Retrain on the whole dataset

    return best_model, best_params, best_score

class ScoringFunction:
    def __init__(self, func, higher_is_better=True):
        self.func = func
        self.higher_is_better = higher_is_better

    def __call__(self, model, X, y):
        return self.func(model, X, y)

    def is_better(self, current_score, best_score):
        if self.higher_is_better:
            return current_score > best_score
        else:
            return current_score < best_score

# Example usage
"""
from sklearn.metrics import accuracy_score

def custom_scoring_function(model, X, y):
    predictions = model.predict(X)
    return accuracy_score(y, predictions)

scoring = ScoringFunction(custom_scoring_function)

model = ModelClass()
param_dist = {
    'param1': randint(1, 4),  # Discrete uniform distribution
    'param2': uniform(0.001, 0.1),  # Continuous uniform distribution
    # Add more parameter distributions
}

best_model, best_params, best_score = random_search(model, param_dist, X, y, scoring, cv=5, n_iter=100, n_jobs=4)
"""
