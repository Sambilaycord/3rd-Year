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




# Split the dataset into training and testing sets (80% train, 20% test)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, random_state = 123, test_size=0.2)

# Train a Logistic Regression model on the training data
lrc = LogisticRegression().fit(X_train2, y_train2)

# Make predictions on the test set
lrc_preds = lrc.predict(X_test2)

# Calculate and display Logistic Regression model accuracy
print(f'Logistic Regression Accuracy: {accuracy_score(y_test2.values, lrc_preds) * 100}%')


# Generate the confusion matrix for the Logistic Regression model
conf_matrix = confusion_matrix(y_test2.values, lrc_preds)


# Generate the classification report for Logistic Regression model
classification_rep = classification_report(y_test2.values, lrc_preds)

# Display the confusion matrix and classification report
print(f"Confusion Matrix: \n{conf_matrix}")
print(f"Classification reprot: \n{classification_rep}")




###### Decision Tree Classifier ###### 
# Train a Decision Tree Classifier on the training data
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state = 123).fit(X_train2, y_train2)


# Make predictions on the test set using the Decision Tree model
dtc_preds = dtc.predict(X_test2)

# Calculate and display Decision Tree Classifier model accuracy
print(f"Decision Trees Classifier Accuracy: {accuracy_score(y_test2.values, dtc_preds) * 100}")


# Generate the confusion matrix for the Decision Tree Classifier model
conf_matrix2 = confusion_matrix(y_test2.values, dtc_preds)

# Generate the classification report for the Decision Tree Classifier model
classification_rep2 = classification_report(y_test2.values, dtc_preds)

# Display the confusion matrix and classification report for the Decision Tree model
print(f"Confusion Matrix: \n {conf_matrix2}")
print(f"Classification Reprot: \n {classification_rep2}")
