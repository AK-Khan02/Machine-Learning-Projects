import numpy as np

def train_test_split(X, y, test_size=0.2, shuffle=True, stratify=None, random_state=None):
    """
    Split dataset into training and test sets.

    Parameters:
    - X : numpy array, dataset features of shape (n_samples, n_features).
    - y : numpy array, target variable of shape (n_samples,).
    - test_size : float, proportion of the dataset to include in the test split.
    - shuffle : boolean, whether to shuffle the data before splitting.
    - stratify : array-like, if not None, data is split in a stratified fashion using this as the class labels.
    - random_state : int, random state for reproducibility.

    Returns:
    - X_train, X_test, y_train, y_test : arrays, split dataset.
    """
    if random_state:
        np.random.seed(random_state)

    # Stratification logic
    if stratify is not None:
        # Ensure y has the same shape as stratify
        if not np.array_equal(np.sort(np.unique(y)), np.sort(np.unique(stratify))):
            raise ValueError("Stratify labels must match dataset labels.")
        
        unique_classes, y_indices = np.unique(stratify, return_inverse=True)
        class_counts = np.bincount(y_indices)
        test_counts = np.round(test_size * class_counts).astype(int)
        train_counts = class_counts - test_counts

        test_indices = np.hstack([np.random.choice(np.where(y_indices==class_idx)[0], size=test_count, replace=False) for class_idx, test_count in enumerate(test_counts)])
        train_indices = np.setdiff1d(np.arange(len(y)), test_indices)
    else:
        # Generate indices for splitting
        indices = np.arange(len(y))
        if shuffle:
            np.random.shuffle(indices)
        
        split_idx = int(len(y) * (1 - test_size))
        train_indices, test_indices = indices[:split_idx], indices[split_idx:]

    # Splitting the dataset
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test
