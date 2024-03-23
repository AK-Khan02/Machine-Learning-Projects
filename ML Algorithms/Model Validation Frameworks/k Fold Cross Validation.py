import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.base import clone

def cross_val_score(model, X, y, cv=5, scoring=None):
    """
    Perform k-fold cross-validation and return a list of scores.

    Parameters:
    - model: The machine learning model to be evaluated.
    - X: Feature matrix of shape (n_samples, n_features).
    - y: Target vector of shape (n_samples,).
    - cv: Number of folds in k-fold cross-validation.
    - scoring: Function to compute the score of each fold. Should take (model, X_test, y_test) as arguments.

    Returns:
    - scores: A list of scores from each fold.
    """
    if scoring is None:
        scoring = accuracy_score  # Default scoring method

    kf = KFold(n_splits=cv, shuffle=True, random_state=None)
    scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Clone the model to ensure each fold gets a fresh model
        cloned_model = clone(model)
        cloned_model.fit(X_train, y_train)
        
        y_pred = cloned_model.predict(X_test)
        score = scoring(y_test, y_pred)
        scores.append(score)

    return scores

# Example usage:
"""
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

# Load a sample dataset
X, y = load_iris(return_X_y=True)

# Choose a model
model = DecisionTreeClassifier()

# Perform cross-validation
scores = cross_val_score(model, X, y, cv=5)

print("Accuracy scores for each fold:", scores)
print("Mean cross-validation accuracy:", np.mean(scores))
"""
