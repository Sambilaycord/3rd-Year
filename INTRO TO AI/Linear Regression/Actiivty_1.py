import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = load_diabetes(as_frame=True)

print("The features are: ", data.feature_names)

# Load the data into a DataFrame
df_d = data.frame

# Create subplots
fig, axes = plt.subplots(2, 5, figsize=(20, 10), sharey=True)  
cols = df_d.columns[:-1] 

# Plot each feature against the target
for i, ax in enumerate(axes.flatten()):  
    ax.scatter(x=df_d[cols[i]], y=df_d['target'])
    ax.set_title(cols[i])  

# Add a label to the figure
fig.text(0.1, 0.5, 'Diabetes Progression', va='center', rotation='vertical')

# Display the plot
#plt.show()


# Features (X) and target (y)
X = df_d.iloc[:, :-1].values
y = df_d.iloc[:, -1]

# Display X and y
print("Features (X):\n", X)
print("Target (y):\n", y)

# Split the data into training and test sets
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state=1)

# Print the shape of the training and test data
print("\nTrain data size: ", X_train1.shape, y_train1.shape)
print("Test data size: ", X_test1.shape, y_test1.shape)

lr_rmodel = LinearRegression().fit(X_train1, y_train1)

print("LR Model Coefficients: ", lr_rmodel.coef_)
print("LR Model Intercept: ", lr_rmodel.intercept_)

mse_lr = mean_squared_error(y_test1, lr_rmodel.predict(X_test1))
print("\nLinear Regression Mean Squared Error is: ", mse_lr)
