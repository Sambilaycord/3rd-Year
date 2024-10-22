import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes

# Load the diabetes dataset
data = load_diabetes(as_frame=True)

print("The features are: ", data.feature_names)

# Load the data into a DataFrame
df_d = data.frame

# Create subplots for visualizing each feature vs target
fig, axes = plt.subplots(2, 5, figsize=(20, 10), sharey=True)  
cols = df_d.columns[:-1] 

# Plot each feature against the target
for i, ax in enumerate(axes.flatten()):  
    ax.scatter(x=df_d[cols[i]], y=df_d['target'])
    ax.set_title(cols[i])  

# Add a label to the figure
fig.text(0.1, 0.5, 'Diabetes Progression', va='center', rotation='vertical')

# Display the plot
plt.show()

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

# Train a Linear Regression model
lr_rmodel = LinearRegression().fit(X_train1, y_train1)

print("Linear Regression Model Coefficients: ", lr_rmodel.coef_)
print("Linear Regression Model Intercept: ", lr_rmodel.intercept_)

# Calculate mean squared error
mse_lr = mean_squared_error(y_test1, lr_rmodel.predict(X_test1))
print("\nLinear Regression Mean Squared Error is: ", mse_lr)

# Select a specific feature (e.g., BMI)
bmi = df_d.iloc[:, 4]
print(X)
print(y)

# Train a simple linear regression model using a single feature (e.g., BMI)
feature = df_d.columns[2]  # Choosing a specific feature like 'bmi'
X_simple = df_d[[feature]].values

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_simple, y, test_size=0.2, random_state=1)

# Correct the usage of the LinearRegression() class
lr_simple_model = LinearRegression().fit(X_train2, y_train2)

# Calculate the mean squared error for simple linear regression
mse_simple = mean_squared_error(y_test2, lr_simple_model.predict(X_test2))
print(f"Simple Linear Regression Mean Squared Error ({feature}): ", mse_simple)

# Plot regression line for the chosen feature
plt.figure(figsize=(10, 6))
sns.regplot(x=df_d[feature], y=df_d['target'], line_kws={"color": "red"})
plt.title(f"Regression Line for {feature}")
plt.ylabel("Diabetes Progression")
plt.xlabel(feature)
plt.show()
