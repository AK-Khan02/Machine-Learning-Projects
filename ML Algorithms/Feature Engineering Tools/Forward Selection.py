from sklearn.base import clone
from sklearn.metrics import mean_squared_error

def forward_selection(estimator, X, y, scoring=mean_squared_error):
    """
    Perform forward feature selection.

    Parameters:
    - estimator: The machine learning model to use.
    - X: Feature matrix.
    - y: Target vector.
    - scoring: Scoring function to evaluate the features.

    Returns:
    - selected_features: List of selected feature indices.
    """
    selected_features = []
    remaining_features = list(range(X.shape[1]))
    current_score = float('inf')

    while remaining_features:
        scores_with_candidates = []
        for feature in remaining_features:
            temp_features = selected_features + [feature]
            model = clone(estimator)
            model.fit(X[:, temp_features], y)
            y_pred = model.predict(X[:, temp_features])
            score = scoring(y, y_pred)
            scores_with_candidates.append((score, feature))
        
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates[0]

        if best_new_score < current_score:
            remaining_features.remove(best_candidate)
            selected_features.append(best_candidate)
            current_score = best_new_score
        else:
            break

    return selected_features
