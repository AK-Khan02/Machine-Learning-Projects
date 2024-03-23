import numpy as np

def sigmoid(z):
    """
    Compute the sigmoid of z.

    Parameters:
    - z : A scalar or numpy array.

    Returns:
    - The sigmoid of z.
    """
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, beta):
    """
    Compute the cost for logistic regression.

    Parameters:
    - X : 2D array where each row represents a training example and each column represents a feature.
    - y : 1D array of labels/target value for each training example.
    - beta : 1D array of fitting parameters.

    Returns:
    - J : The cost of using beta as the parameter for logistic regression to fit the data points in X and y.
    """
    m = len(y)
    h = sigmoid(X.dot(beta))
    epsilon = 1e-5  # to avoid log(0)
    J = -1/m * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
    return J

def gradient_descent(X, y, beta, alpha, iterations):
    """
    Perform gradient descent to learn beta.

    Parameters:
    - X : 2D array of our training data, where each row represents a training example and each column represents a feature.
    - y : 1D array of the target variable for each training example.
    - beta : 1D array of initial fitting parameters.
    - alpha : Learning rate.
    - iterations : Number of iterations to run gradient descent.

    Returns:
    - beta : The final values of parameters.
    - J_history : A list containing the cost value for each iteration, useful for debugging.
    """
    m = len(y)
    J_history = []

    for i in range(iterations):
        h = sigmoid(X.dot(beta))
        gradient = np.dot(X.T, (h - y)) / m
        beta -= alpha * gradient
        J_history.append(compute_cost(X, y, beta))

    return beta, J_history

# Example usage:
# Assuming X_train is your input features and y_train is the binary target variable
# Add intercept term to X_train
X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
beta_initial = np.zeros(X_train.shape[1])
alpha = 0.01  # Learning rate
iterations = 1000

# Perform gradient descent
beta, J_history = gradient_descent(X_train, y_train, beta_initial, alpha, iterations)
