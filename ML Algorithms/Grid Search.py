from itertools import product
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import numpy as np

def cross_val_score(model, X, y, scoring, cv):
    kf = KFold(n_splits=cv)
    scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model.fit(X_train, y_train)
        scores.append(scoring(model, X_test, y_test))

    return np.mean(scores)

def evaluate_model(model, params, X, y, scoring, cv):
    model.set_params(**params)
    score = cross_val_score(model, X, y, scoring, cv)
    return score, params

def grid_search_cv(model, param_grid, X, y, scoring, cv=5, n_jobs=-1, early_stopping=None):
    best_score = float('-inf') if scoring.higher_is_better else float('inf')
    best_params = None
    no_improvement_count = 0

    param_combinations = list(product(*param_grid.values()))
    
    # Parallel execution of model evaluation
    results = Parallel(n_jobs=n_jobs)(delayed(evaluate_model)(model, dict(zip(param_grid.keys(), params)), X, y, scoring, cv) for params in param_combinations)

    for score, params in results:
        if scoring.is_better(score, best_score):
            best_score = score
            best_params = params
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if early_stopping and no_improvement_count >= early_stopping:
            print("Early stopping triggered.")
            break

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
param_grid = {
    'param1': [1, 2, 3],
    'param2': [0.1, 0.01, 0.001],
    # Add more parameters
}

best_model, best_params, best_score = grid_search_cv(model, param_grid, X, y, scoring, cv=5, n_jobs=4, early_stopping=10)
"""
