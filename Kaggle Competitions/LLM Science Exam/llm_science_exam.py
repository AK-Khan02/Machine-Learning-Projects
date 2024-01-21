# Link to Kaggle Competition: https://www.kaggle.com/competitions/kaggle-llm-science-exam

# Importing Relevant Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier

import matplotlib.pyplot as plt

data = pd.read_csv('train.csv')

# Combine prompt with each answer choice
combined_features = []
for index, row in data.iterrows():
    for option in ['A', 'B', 'C', 'D', 'E']:
        combined_text = row['prompt'] + ' ' + row[option]
        combined_features.append((combined_text, row['answer'] == option))

# Convert the list to a DataFrame
combined_df = pd.DataFrame(combined_features, columns=['text', 'is_correct'])

# Applying TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(combined_df['text'])
y = combined_df['is_correct']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Machine Learning Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": MultinomialNB(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Apply Models and Compare
results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    results[model_name] = accuracy

# Plot Results
labels, values = zip(*results.items())

plt.figure(figsize=(15, 10))
plt.bar(range(len(results)), values, tick_label=labels)
plt.xticks(rotation=45)
plt.xlabel('ML Model')
plt.ylabel('Accuracy')
plt.title('Comparison of ML Models')
plt.show()

# Print Results Numerically
print("Model Accuracy Results:")
for model_name, accuracy in results.items():
    print(f"{model_name}: {accuracy:.4f}")  # Prints accuracy to four decimal places

# Hyperparameter Tuning for SVM
parameters = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01]}
svc = SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(X_train, y_train)
best_svc = clf.best_estimator_

# Replace the default SVM model with the optimized one
models['SVM'] = best_svc

# Apply Models and Compare
results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    results[model_name] = accuracy

# Print Results Numerically
print("Model Accuracy Results:")
for model_name, accuracy in results.items():
    print(f"{model_name}: {accuracy:.4f}")  # Prints accuracy to four decimal places

# Apply Models and Compare
results = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    results[model_name] = accuracy

# Plot Results
labels, values = zip(*results.items())

plt.figure(figsize=(15, 10))
plt.bar(range(len(results)), values, tick_label=labels)
plt.xticks(rotation=45)
plt.xlabel('ML Model')
plt.ylabel('Accuracy')
plt.title('Comparison of ML Models')
plt.show()

