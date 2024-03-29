import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt

# Load the dataset
creditcard_data = pd.read_csv("creditcard.csv")

# Basic Exploration
print(creditcard_data.shape)
print(creditcard_data.head(6))
print(creditcard_data.tail(6))
print(creditcard_data['Class'].value_counts())
print(creditcard_data['Amount'].describe())
print(creditcard_data.columns)
print(creditcard_data['Amount'].var())
print(creditcard_data['Amount'].std())

# Feature Scaling
creditcard_data['Amount'] = (creditcard_data['Amount'] - creditcard_data['Amount'].mean()) / creditcard_data['Amount'].std()

# Prepare the data
NewData = creditcard_data.drop(columns=['Time'])  # Assuming 'Time' is the first column
print(NewData.head())

# Splitting the data
X = NewData.drop('Class', axis=1)
y = NewData['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=123)

# Logistic Regression Model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
lr_predictions = logistic_model.predict_proba(X_test)[:,1]

# Plotting ROC Curve for Logistic Regression
fpr, tpr, thresholds = roc_curve(y_test, lr_predictions)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='blue', label=f'Logistic Regression AUC = {roc_auc:.2f}')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

# Decision Tree Model
decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(X_train, y_train)
dt_predictions = decision_tree_model.predict_proba(X_test)[:,1]

# Neural Network Model
ann_model = MLPClassifier()
ann_model.fit(X_train, y_train)
ann_predictions = ann_model.predict_proba(X_test)[:,1]

# Light GBM Model
lgbm_model = LGBMClassifier(n_estimators=500, learning_rate=0.01, max_depth=3, num_leaves=31, min_data_in_leaf=100, subsample=0.5)
lgbm_model.fit(X_train, y_train)
lgbm_predictions = lgbm_model.predict_proba(X_test)[:, 1]

# Plotting ROC Curve for LightGBM
fpr_lgbm, tpr_lgbm, thresholds_lgbm = roc_curve(y_test, lgbm_predictions)
roc_auc_lgbm = auc(fpr_lgbm, tpr_lgbm)
plt.figure(figsize=(8, 6))
plt.plot(fpr_lgbm, tpr_lgbm, color='red', lw=2, label=f'LGBM AUC = {roc_auc_lgbm:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - LightGBM')
plt.legend(loc='lower right')
plt.show()

# Print AUC values
print(f'Logistic Regression AUC: {roc_auc:.2f}')
print(f'LightGBM AUC: {roc_auc_lgbm:.2f}')

