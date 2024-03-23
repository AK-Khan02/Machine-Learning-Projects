def recursive_feature_elimination(estimator, X, y, n_features_to_select=None):
    """
    Perform recursive feature elimination.

    Parameters:
    - estimator: The machine learning model to use.
    - X: Feature matrix.
    - y: Target vector.
    - n_features_to_select: Number of features to select.

    Returns:
    - selected_features: List of selected feature indices.
    """
    if n_features_to_select is None:
        n_features_to_select = X.shape[1] // 2

    selected_features = list(range(X.shape[1]))
    current_num_features = X.shape[1]

    while current_num_features > n_features_to_select:
        scores = []

        for feature in selected_features:
            temp_features = list(selected_features)
            temp_features.remove(feature)
            model = clone(estimator)
            model.fit(X[:, temp_features], y)
            y_pred = model.predict(X[:, temp_features])
            score = mean_squared_error(y, y_pred)
            scores.append((score, feature))

        scores.sort(reverse=True)
        worst_feature = scores[0][1]
        selected_features.remove(worst_feature)
        current_num_features -= 1

    return selected_features
