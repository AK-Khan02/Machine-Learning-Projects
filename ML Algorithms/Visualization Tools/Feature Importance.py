import matplotlib.pyplot as plt
import numpy as np

# The function assumes that 'model' has a '.feature_importance_' attirbute, which is true of many models
def plot_feature_importance(model, feature_names, title="Feature Importance", figsize=(10, 6)):
    """
    Visualize the importance of each feature in the model.

    Parameters:
    - model: The fitted model with a feature_importances_ attribute.
    - feature_names: A list of names for each feature.
    - title: Title of the plot.
    - figsize: Size of the figure (width, height).
    """
    # Ensure the model has been fitted
    if not hasattr(model, 'feature_importances_'):
        raise ValueError("Model does not have feature_importances_ attribute.")

    feature_importances = model.feature_importances_
    indices = np.argsort(feature_importances)[::-1]

    # Create the plot
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.bar(range(len(feature_importances)), feature_importances[indices], align="center", color='skyblue', alpha=0.7)
    plt.xticks(range(len(feature_importances)), [feature_names[i] for i in indices], rotation=90)
    plt.xlim([-1, len(feature_importances)])
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()  # Adjust layout to not cut off labels

    plt.show()

# Example usage:
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load data
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Plot feature importance
plot_feature_importance(model, feature_names)
"""
