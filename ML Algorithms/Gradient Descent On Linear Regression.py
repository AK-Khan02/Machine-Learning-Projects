import numpy as np

def feature_scaling(X):
    """
    Perform feature scaling using standardization.

    Parameters:
    - X : 2D array of features.

    Returns:
    - X_scaled : 2D array of standardized features.
    - mean : 1D array of means of each feature.
    - std : 1D array of standard deviations of each feature.
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_scaled = (X - mean) / std
    return X_scaled, mean, std

def compute_cost(X, y, beta, lambda_):
    """
    Compute the regularized cost function for linear regression.

    Parameters:
    - X : 2D array of features.
    - y : 1D array of target variable.
    - beta : 1D array of fitting parameters.
    - lambda_ : Regularization parameter.

    Returns:
    - J : Regularized cost.
    """
    m = len(y)
    J = np.sum((X.dot(beta) - y) ** 2) / (2 * m) + (lambda_ / (2 * m)) * np.sum(beta[1:] ** 2)
    return J

def gradient_descent(X, y, beta, alpha, iterations, lambda_, batch_size):
    """
    Perform mini-batch gradient descent with regularization.

    Parameters:
    - X : 2D array of features.
    - y : 1D array of target variable.
    - beta : 1D array of initial fitting parameters.
    - alpha : Learning rate.
    - iterations : Number of iterations.
    - lambda_ : Regularization parameter.
    - batch_size : Size of the mini-batch.

    Returns:
    - beta : Updated parameters.
    - J_history : Cost history over iterations.
    """
    m, n = X.shape
    J_history = []

    for iteration in range(iterations):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        for i in range(0, m, batch_size):
            Xi = X_shuffled[i:i + batch_size]
            yi = y_shuffled[i:i + batch_size]
            
            prediction = Xi.dot(beta)
            errors = prediction - yi
            delta = Xi.T.dot(errors) / batch_size
            delta[1:] += (lambda_ / m) * beta[1:]  # Regularization, skip beta[0]

            beta -= alpha * delta

        J_history.append(compute_cost(X, y, beta, lambda_))

    return beta, J_history

# Example usage
# Assuming X_train is your input matrix of features and y_train is the target variable
X_train_scaled, mean, std = feature_scaling(X_train)
X_train_scaled = np.hstack((np.ones((X_train_scaled.shape[0], 1)), X_train_scaled))  # Add intercept term

beta_initial = np.zeros(X_train_scaled.shape[1])
alpha = 0.01  # Learning rate
iterations = 1000
lambda_ = 1  # Regularization parameter
batch_size = 32  # Mini-batch size

beta, J_history = gradient_descent(X_train_scaled, y_train, beta_initial, alpha, iterations, lambda_, batch_size)
