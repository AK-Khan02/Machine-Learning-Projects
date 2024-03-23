import numpy as np

def pca(X, n_components=None):
    """
    Perform PCA on dataset X to reduce its dimensionality.

    Parameters:
    - X : numpy array of shape (n_samples, n_features).
    - n_components : int or None, the number of principal components to compute.

    Returns:
    - X_pca : The transformed data matrix of shape (n_samples, n_components).
    """
    # Standardize the data
    X_centered = X - np.mean(X, axis=0)

    # Compute the covariance matrix
    covariance_matrix = np.cov(X_centered, rowvar=False)

    # Compute eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort eigenvectors by eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Select the top n_components eigenvectors
    if n_components is not None:
        eigenvectors = eigenvectors[:, :n_components]

    # Transform the data
    X_pca = np.dot(X_centered, eigenvectors)

    return X_pca
