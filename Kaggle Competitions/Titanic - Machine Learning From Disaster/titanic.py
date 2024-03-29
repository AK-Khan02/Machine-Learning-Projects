# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
from collections import Counter

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# validation
from sklearn.metrics import accuracy_score

encoder = LabelEncoder()
scaler = StandardScaler()

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
#fake_submit = pd.read_csv("gender_submission.csv")
combine = [train_data, test_data]

# Outlier detection

def detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []

    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1

        # outlier step
        outlier_step = 1.5 * IQR

        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        # append the found outlier indices for col to the list of outlier indices
        outlier_indices.extend(outlier_list_col)

    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    return multiple_outliers

# detect outliers from Age, SibSp , Parch and Fare
Outliers_to_drop = detect_outliers(train_data,2,["Age","SibSp","Parch","Fare"])
train_data = train_data.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)

train_data.head()

"""# Analyze

phân tích dựa trên những feature có sẵn, chưa tạo mới các feature khác

## Basic Analyze
"""

train_data.info()

train_data.describe(percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99])
# 1% la sibsp > 5
# 1% age > 65 ; 50% thuoc nhom [20,40]
# 38% survived -> 62% dead

train_data.describe(include=["O"])

train_data.corr().sort_values(by="Survived")

## Make assumtions

### Complete : Age, Embarked
### Convert: Sex, Ticket, Embarked
### Creating: Age Bands, Fare range, SibSp&Parch
###

"""## Analyze by pivot point"""

train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

train_data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

train_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

train_data[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)

train_data[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

"""## Visualize Analyze"""

## Age vs Survived

g = sns.FacetGrid(train_data, col='Survived')
g.map(plt.hist, 'Age', bins=20)

## Age vs Survived | Pclass

# grid = sns.FacetGrid(train_data, col='Pclass', hue='Survived')
grid = sns.FacetGrid(train_data, col='Survived', row='Pclass', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();

## Pclass vs Sex vs Survived | Embarked
grid = sns.FacetGrid(train_data, row='Embarked', height=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()

## Fare vs Survived | Embarked vs Sex
grid = sns.FacetGrid(train_data, row='Embarked', col='Sex', height=2.2, aspect=1.6)
grid.map(sns.barplot, 'Survived', 'Fare', alpha=.5, ci=None)
grid.add_legend()

"""# Wrangle data"""

train_data = train_data.drop(["Cabin", "PassengerId", "Name"], axis = 1)
test_data = test_data.drop(["Cabin", "Name"], axis = 1)
combine = [train_data, test_data]

for dataset in combine:
    dataset["Sex"] = encoder.fit_transform(dataset["Sex"])

freq_port = train_data["Embarked"].dropna().mode()
for dataset in combine:
    dataset["Embarked"] = dataset["Embarked"].fillna(freq_port[0])
    dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].mean())

for dataset in combine:
    dataset["Embarked"] = encoder.fit_transform(dataset["Embarked"])

for dataset in combine:
    dataset["Family Size"] = dataset["SibSp"] + dataset["Parch"] + 1

for dataset in combine:
    dataset["Alone"] = 0
    dataset.loc[dataset["Family Size"] == 1, "Alone"] = 1

train_data = train_data.drop(["Family Size", "SibSp", "Parch"], axis = 1)
test_data = test_data.drop(["Family Size", "SibSp", "Parch"], axis = 1)
combine = [train_data, test_data]

grid = sns.FacetGrid(train_data, col='Pclass', row='Alone', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();

age_guess = np.zeros((train_data["Alone"].unique().size, train_data["Pclass"].unique().size))

age_guess.shape

for dataset in combine:
    for i in range(age_guess.shape[0]):
        for j in range(age_guess.shape[1]):
            guess_df = dataset.loc[(dataset["Alone"] == i) & (dataset["Pclass"] == j + 1), "Age"].dropna()
            guess = guess_df.mean()
            age_guess[i,j] = guess
    for i in range(age_guess.shape[0]):
        for j in range(age_guess.shape[1]):
            dataset.loc[(dataset["Age"].isnull()) & (dataset["Alone"] == i) & (dataset["Pclass"] == j + 1), "Age"] = age_guess[i,j]
    dataset["Age"] = np.round(dataset["Age"])

grid = sns.FacetGrid(train_data, col='Survived', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();

train_data["AgeBand"] = pd.cut(train_data["Age"], 5)
train_data[["AgeBand", "Survived"]].groupby("AgeBand", as_index=False).mean().sort_values(by="AgeBand")

for dataset in combine:
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']

train_data = train_data.drop("Ticket", axis=1)
test_data = test_data.drop("Ticket", axis=1)
combine = [train_data, test_data]

train_data['FareBand'] = pd.qcut(train_data['Fare'], 4)
train_data[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3

train_data = train_data.drop(["AgeBand", "FareBand"], axis = 1)

"""# Model"""

X = train_data.drop("Survived", axis=1)
y = train_data["Survived"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_test = test_data.drop("PassengerId", axis=1)

logreg = LogisticRegression(solver="lbfgs")
logreg.fit(X_train, y_train)
log_pred = logreg.predict(X_val)
print("Accuracy score:", accuracy_score(log_pred, y_val))
print("Train Accuracy: ", accuracy_score(logreg.predict(X_train), y_train))

svc = SVC(kernel="rbf", C=0.9, gamma=0.2)
svc.fit(X_train, y_train)
svc_pred = svc.predict(X_val)
print("Val Accuracy: ", accuracy_score(svc_pred, y_val))
print("Train Accuracy: ", accuracy_score(svc.predict(X_train), y_train))

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_val)
print("Val Accuracy: ", accuracy_score(knn_pred, y_val))
print("Train Accuracy: ", accuracy_score(knn.predict(X_train), y_train))

tree = DecisionTreeClassifier(max_depth=3, criterion="entropy")
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_val)
print("Val Accuracy: ", accuracy_score(tree_pred, y_val))
print("Train Accuracy: ", accuracy_score(tree.predict(X_train), y_train))

forest = RandomForestClassifier(n_estimators=50, criterion="entropy", max_depth=5)
forest.fit(X_train, y_train)
forest_pred = forest.predict(X_val)
print("Val Accuracy: ", accuracy_score(forest_pred, y_val))
print("Train Accuracy: ", accuracy_score(forest.predict(X_train), y_train))

"""# Submit"""

y_test = forest.predict(X_test)
submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": y_test
})
submission.to_csv('submission.csv', index=False)



