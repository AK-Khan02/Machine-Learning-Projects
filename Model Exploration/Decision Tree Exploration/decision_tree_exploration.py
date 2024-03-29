# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

C# Load data
beans = pd.read_csv("beans.csv")
beans.head()

X = beans.drop("Class", axis = 1)
y = beans.Class
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 546)

"""# Creating a Basic Decision Tree"""

# Create and fit decision tree
to_plot = DecisionTreeClassifier(max_depth = 4)
to_plot.fit(X_train, y_train)

# Plot decision tree
plt.figure(figsize = (50,8))
plot_tree(to_plot, filled = True, feature_names = X.columns.values)

simple = DecisionTreeClassifier(random_state = 546)
simple.fit(X_train, y_train)

y_pred_simple = simple.predict(X_test)
ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test, y_pred_simple)).plot()
plt.title("Simple Decision Tree")
print(classification_report(y_test, y_pred_simple))

"""# Pruning"""

# Obtain pruning path
path = simple.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

trees = []

# Test various levels of alpha
# This takes about a minute, there are over 300 alpha values to train for
for ccp_alpha in ccp_alphas:
    tree = DecisionTreeClassifier(random_state = 546, ccp_alpha = ccp_alpha)
    tree.fit(X_train, y_train)
    trees.append(tree)

# Plot alpha vs nodes, depth
# We are using a logarithmnic scale for alpha because if we didnt, a lot of the
# values would be squished to the left size of the graph
log_ccp_alphas = np.log10(ccp_alphas)
fig, ax = plt.subplots(2, 1)

node_counts = [tree.tree_.node_count for tree in trees]
ax[0].plot(log_ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
ax[0].set_xlabel("log(alpha)")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")

depth = [tree.tree_.max_depth for tree in trees]
ax[1].plot(log_ccp_alphas, depth, marker="o", drawstyle="steps-post")
ax[1].set_xlabel("log(alpha)")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")

fig.tight_layout()

# Plot alpha vs performance
train_scores = [tree.score(X_train, y_train) for tree in trees]
test_scores = [tree.score(X_test, y_test) for tree in trees]

fig, ax = plt.subplots()
ax.set_xlabel("log(alpha)")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(log_ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
ax.plot(log_ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
ax.legend()
plt.show()

"""# Bagging"""

bagging = BaggingClassifier(random_state = 546) # Defaults to 10 trees
bagging.fit(X_train, y_train)

y_pred_bagging = bagging.predict(X_test)
ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test, y_pred_bagging)).plot()
plt.title("Bagged Trees")
print(classification_report(y_test, y_pred_simple)) # For comparison
print(classification_report(y_test, y_pred_bagging))

# Vary number of trees
n_trees = [1, 10, 25, 50, 75, 100]
bag_train_scores = []
bag_test_scores = []

# Takes around half a minute
for n in n_trees:
    bag = BaggingClassifier(n_estimators = n, random_state = 546)
    bag.fit(X_train, y_train)
    bag_train_scores.append(bag.score(X_train, y_train))
    bag_test_scores.append(bag.score(X_test, y_test))

# Plot performances
fig, ax = plt.subplots()
ax.set_xlabel("number of trees")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs number of trees for training and testing sets")
ax.plot(n_trees, bag_train_scores, marker="o", label="train")
ax.plot(n_trees, bag_test_scores, marker="o", label="test")
ax.legend()
plt.show()

# Vary size of bootstrap sample
sample_size = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
bag_train_scores = []
bag_test_scores = []

for size in sample_size:
    bag = BaggingClassifier(max_samples = size, random_state = 546)
    bag.fit(X_train, y_train)
    bag_train_scores.append(bag.score(X_train, y_train))
    bag_test_scores.append(bag.score(X_test, y_test))

# Plot performances
fig, ax = plt.subplots()
ax.set_xlabel("sample size")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs sample size for training and testing sets")
ax.plot(sample_size, bag_train_scores, marker="o", label="train")
ax.plot(sample_size, bag_test_scores, marker="o", label="test")
ax.legend()
plt.show()

"""# Random Subspace"""

# Vary number of features
n_features = range(2,17)
bag_train_scores = []
bag_test_scores = []

for n in n_features:
    bag = BaggingClassifier(max_features = n, bootstrap = False, random_state = 546)
    bag.fit(X_train, y_train)
    bag_train_scores.append(bag.score(X_train, y_train))
    bag_test_scores.append(bag.score(X_test, y_test))

# Plot performances
fig, ax = plt.subplots()
ax.set_xlabel("number of features")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs number of features for training and testing sets")
ax.plot(n_features, bag_train_scores, marker="o", label="train")
ax.plot(n_features, bag_test_scores, marker="o", label="test")
ax.legend()
plt.show()

"""# Random Forest"""

forest = RandomForestClassifier(random_state = 546) # Defaults to 100 trees, sqrt of number of features
forest.fit(X_train, y_train)

y_pred_forest = forest.predict(X_test)
ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test, y_pred_forest)).plot()
plt.title("Random Forest")
print(classification_report(y_test, y_pred_simple)) # For comparison
print(classification_report(y_test, y_pred_forest))

"""# Boosting"""

boost = GradientBoostingClassifier(random_state = 546)
boost.fit(X_train, y_train)

y_pred_boost = boost.predict(X_test)
ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test, y_pred_boost)).plot()
plt.title("Gradient Boosted Tree")
print(classification_report(y_test, y_pred_simple)) # For comparison
print(classification_report(y_test, y_pred_boost))

# Vary learning rate
learning_rate = range(-4, 1)
bag_test_scores = []

# Takes around 3 minutes
for rate in learning_rate:
    bag = GradientBoostingClassifier(learning_rate = 10 ** rate, random_state = 546)
    bag.fit(X_train, y_train)
    bag_test_scores.append(bag.score(X_test, y_test))

# Plot performances
fig, ax = plt.subplots()
ax.set_xlabel("log(learning rate)")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs learning rate for testing sets")
ax.plot(learning_rate, bag_test_scores, marker="o", label="test")
ax.legend()
plt.show()
