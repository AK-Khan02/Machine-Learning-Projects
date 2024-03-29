# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.neighbors import KNeighborsClassifier, NeighborhoodComponentsAnalysis
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load data
X, y = load_wine(return_X_y = True, as_frame = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 546)

# Standardize features
scaler = StandardScaler().set_output(transform = "pandas")
X_scaled = scaler.fit_transform(X)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Preview the raw data
X.head()

"""# Creating a Basic KNN Classifier"""

# Train the model
knn = KNeighborsClassifier() # Defaults to k = 5
knn.fit(X_train_scaled, y_train)

# Generate predictions on the test set
y_pred = knn.predict(X_test_scaled)
ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test, y_pred)).plot()
plt.title("KNN (k = 5)")

"""# Distance Metrics"""

# Train the model
manhattan = KNeighborsClassifier(p = 1)
manhattan.fit(X_train_scaled, y_train)

# Generate predictions on the test set
y_pred_manhattan = manhattan.predict(X_test_scaled)
ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test, y_pred_manhattan)).plot()
plt.title("Manhattan Distance KNN (k = 5)")

# Train the model
knn_weighted = KNeighborsClassifier(weights = "distance")
knn_weighted.fit(X_train_scaled, y_train)

# Generate predictions on the test set
y_pred_weighted = knn_weighted.predict(X_test_scaled)
ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test, y_pred_weighted)).plot()
plt.title("Weighted KNN (k = 5)")

"""# Dimensionality Reduction

"""

# Create and fit PCA model
pca = PCA(n_components = 2, random_state = 546)
pca.fit(X_scaled)

# Plot reduced dimensions
X_scaled_pca = pca.transform(X_scaled)
plt.scatter(X_scaled_pca[:, 0], X_scaled_pca[:, 1], c = y)
plt.title("PCA, KNN (k = 5)")

# Implement KNN with PCA dimensionality reduction
knn_pca = KNeighborsClassifier()
knn_pca.fit(pca.transform(X_train_scaled), y_train)

# Generate predictions on the test set
y_pred_pca = knn_pca.predict(pca.transform(X_test_scaled))
plt.figure()
ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test, y_pred_pca)).plot()
plt.title("PCA, KNN (k = 5)")

# Create and fit LDA model
lda = LinearDiscriminantAnalysis(n_components = 2)
lda.fit(X_scaled, y) # Note: LDA needs the class values

# Plot reduced dimensions
X_scaled_lda = lda.transform(X_scaled)
plt.scatter(X_scaled_lda[:, 0], X_scaled_lda[:, 1], c = y)
plt.title("LDA, KNN (k = 5)")

# Implement KNN with LDA dimensionality reduction
knn_lda = KNeighborsClassifier()
knn_lda.fit(lda.transform(X_train_scaled), y_train)

# Generate predictions on the test set
y_pred_lda = knn_lda.predict(lda.transform(X_test_scaled))
plt.figure()
ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test, y_pred_lda)).plot()
plt.title("LDA, KNN (k = 5)")

# Create and fit NCA model
nca = NeighborhoodComponentsAnalysis(n_components = 2, random_state = 546)
nca.fit(X_scaled, y) # Note: NCA needs the class values

# Plot reduced dimensions
X_scaled_nca = nca.transform(X_scaled)
plt.scatter(X_scaled_nca[:, 0], X_scaled_nca[:, 1], c = y)
plt.title("NCA, KNN (k = 5)")

# Implement KNN with NCA dimensionality reduction
knn_nca = KNeighborsClassifier()
knn_nca.fit(nca.transform(X_train_scaled), y_train)

# Generate predictions on the test set
y_pred_nca = knn_nca.predict(nca.transform(X_test_scaled))
plt.figure()
ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test, y_pred_nca)).plot()
plt.title("NCA, KNN (k = 5)")

"""# Feature Scaling

Let's start by looking at the raw data and the scaled data.
"""

X.describe().round(2)

X_scaled.describe().round(2)

# Extract proline and hue from data
X_plot = X[["proline", "hue"]]
X_plot_scaled = scaler.fit_transform(X_plot)

# Wrapper function
def fit_and_plot_model(X_plot, y, clf, ax):
    clf.fit(X_plot, y)
    disp = DecisionBoundaryDisplay.from_estimator(
        clf,
        X_plot,
        response_method="predict",
        alpha=0.5,
        ax=ax,
    )
    disp.ax_.scatter(X_plot["proline"], X_plot["hue"], c=y, s=20, edgecolor="k")
    disp.ax_.set_xlim((X_plot["proline"].min(), X_plot["proline"].max()))
    disp.ax_.set_ylim((X_plot["hue"].min(), X_plot["hue"].max()))
    return disp.ax_

# Create plots of decision boundaries
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))

fit_and_plot_model(X_plot, y, knn, ax1)
ax1.set_title("KNN without scaling")

fit_and_plot_model(X_plot_scaled, y, knn, ax2)
ax2.set_xlabel("scaled proline")
ax2.set_ylabel("scaled hue")
_ = ax2.set_title("KNN with scaling")
