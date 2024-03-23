# Import libraries
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns

# Load dataset
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
pima = pd.read_csv("diabetes.csv", header=1, names=col_names)

# Split dataset into features and target variable
feature_cols = ['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree', 'skin']
X = pima[feature_cols]
y = pima.label

sns.pairplot(pima) # This line takes a while to run

# See existing data types
print(type(X))
print(type(y))

# Convert to PyTorch data types
X = torch.tensor(X.to_numpy(), dtype=torch.float32)
y = torch.tensor(y.to_numpy(), dtype=torch.float32).reshape(-1, 1)

# See new data types
print(type(X))
print(type(y))

from sklearn.model_selection import train_test_split

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=16) # Set random_state for reproducibility

model = nn.Sequential(
    nn.Linear(8, 12), # 8 input nodes to 12 nodes in the first hidden layer
    nn.ReLU(),        # ReLU activation for first hidden layer
    nn.Linear(12, 8), # 12 nodes from first hidden layer to 8 in the second
    nn.ReLU(),        # ReLU activation for second hidden layer
    nn.Linear(8, 1),  # 8 nodes from second hidden layer to 1 output node
    nn.Sigmoid()      # Sigmoid activation for output layer
)

print(model) # Check the model

loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

n_epochs = 100
batch_size = 10

for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i+batch_size]
        y_pred = model(Xbatch)
        ybatch = y[i:i+batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Finished epoch {epoch}, latest loss {loss}')

with torch.no_grad(): # This prevents the model from calculating gradients when we don't need it to (optional)
  y_pred = model(X_test)

print(y_pred)

"""Our model outputs probabilities between 0 and 1. We can convert these probabilities to binary predictions by rounding them."""

y_pred = y_pred.round()

from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

class_names=[0,1] # Names of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# Create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
