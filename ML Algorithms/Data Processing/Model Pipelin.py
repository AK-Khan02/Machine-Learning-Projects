import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Sample dataset
data = pd.DataFrame({
    'feature1': np.random.randn(100),
    'feature2': np.random.rand(100),
    'target': np.random.choice([0, 1], 100)
})

# 1. Data Cleaning
## Handling missing values
data.fillna(data.mean(), inplace=True)

## Removing outliers
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
data = data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).any(axis=1)]

# 2. Feature Engineering
## Creating a new feature as an example
data['feature3'] = data['feature1'] ** 2 + data['feature2']

# Splitting the dataset
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Normalization
scaler = StandardScaler()

# 4. Model Selection
model = RandomForestClassifier(random_state=42)

# 5. Hyperparameter Tuning
pipeline = Pipeline([('scaler', scaler), ('model', model)])
param_grid = {
    'model__n_estimators': [10, 50, 100],
    'model__max_depth': [None, 10, 20, 30]
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_

# 6. Evaluation
## Cross-validation on the training set
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
print("CV average score: %.2f" % cv_scores.mean())

## Test set evaluation
y_pred = best_model.predict(X_test)
test_score = accuracy_score(y_test, y_pred)
print("Test set score: %.2f" % test_score)
