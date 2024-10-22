import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the mushroom dataset
df_mushroom = pd.read_csv('mushroom_cleaned.csv')
df_mushroom.head()      #Display the first few rows of the dataset
df_mushroom.describe()  #Display the statistical summary of the dataset


###### Logistic Regression ###### 
# Rename 'class' column to 'mushroom_class' to avoid conflict with Python keywords
df_mushroom.rename(columns={'class': 'mushroom_class'}, inplace=True)   


# Define feature set X and target variable y
X = df_mushroom.drop(['mushroom_class'], axis=1)    # All columns except 'mushroom_class'
y = df_mushroom.mushroom_class                      # Target Variable


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


# Interpretation of Results
# Comparing the performance of Logistic Regression and Decision Tree Classifier
print("\nInterpretation of Results")
print("The Decision Tree Classifier performs better than the Logistic Regression model, with an accuracy that is about 33.95% higher (97.59% vs 68.75%).")
print("This suggest that the decision tree is able to capture more patterns in the data that lead to correct classifications")


