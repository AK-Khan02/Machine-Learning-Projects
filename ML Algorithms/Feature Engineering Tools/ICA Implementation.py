from sklearn.decomposition import FastICA

def ica(X, n_components=None):
    """
    Perform ICA on dataset X to find independent components.

    Parameters:
    - X : numpy array of shape (n_samples, n_features).
    - n_components : int or None, the number of components to compute.

    Returns:
    - X_ica : The transformed data matrix of shape (n_samples, n_components).
    """
    ica = FastICA(n_components=n_components, random_state=0)
    X_ica = ica.fit_transform(X)

    return X_ica
