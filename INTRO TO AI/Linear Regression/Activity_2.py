import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier

# Load the mushroom dataset
df_mushroom = pd.read_csv('mushroom_cleaned.csv')
df_mushroom.head()      # Display the first few rows of the dataset
df_mushroom.describe()  # Display the statistical summary of the dataset

###### Logistic Regression ######
# Rename 'class' column to 'mushroom_class' to avoid conflict with Python keywords
df_mushroom.rename(columns={'class': 'mushroom_class'}, inplace=True)

# Define feature set X and target variable y
X = df_mushroom.drop(['mushroom_class'], axis=1)    # All columns except 'mushroom_class'
y = df_mushroom.mushroom_class                      # Target Variable

# Split the dataset into training and testing sets (80% train, 20% test)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, random_state=123, test_size=0.2)

# Train a Logistic Regression model on the training data
lrc = LogisticRegression().fit(X_train2, y_train2)

# Make predictions on the test set
lrc_preds = lrc.predict(X_test2)

# Calculate and store Logistic Regression model accuracy
lrc_accuracy = accuracy_score(y_test2.values, lrc_preds) * 100
print(f'Logistic Regression Accuracy: {lrc_accuracy}% \n')


# Generate the confusion matrix and classification report for Logistic Regression
conf_matrix = confusion_matrix(y_test2.values, lrc_preds)
classification_rep = classification_report(y_test2.values, lrc_preds)

# Display the confusion matrix and classification report
print(f"Confusion Matrix: \n{conf_matrix}")
print(f"Classification report: \n{classification_rep}")


###### Decision Tree Classifier ######
# Train a Decision Tree Classifier on the training data
dtc = DecisionTreeClassifier(random_state=123).fit(X_train2, y_train2)

# Make predictions on the test set using the Decision Tree model
dtc_preds = dtc.predict(X_test2)

# Calculate and store Decision Tree Classifier model accuracy
dtc_accuracy = accuracy_score(y_test2.values, dtc_preds) * 100
print(f"Decision Tree Classifier Accuracy: {dtc_accuracy}%")

# Generate the confusion matrix and classification report for the Decision Tree model
conf_matrix2 = confusion_matrix(y_test2.values, dtc_preds)
classification_rep2 = classification_report(y_test2.values, dtc_preds)

# Display the confusion matrix and classification report for the Decision Tree model
print(f"Confusion Matrix: \n{conf_matrix2}")
print(f"Classification Report: \n{classification_rep2}")


###### Interpretation of Results ######
accuracy_diff = dtc_accuracy - lrc_accuracy

print("\nInterpretation of Results")
if lrc_accuracy > dtc_accuracy:
    print(f"The Logistic Regression model has a higher accuracy ({lrc_accuracy:.2f}%) than the Decision Tree Classifier ({dtc_accuracy:.2f}%).")
    print("This indicates that the Logistic Regression model performed better in this scenario.")
elif lrc_accuracy < dtc_accuracy:
    print(f"The Decision Tree Classifier has a higher accuracy ({dtc_accuracy:.2f}%) compared to the Logistic Regression model ({lrc_accuracy:.2f}%).")
    print("This suggests that the Decision Tree is better at capturing patterns in the data.")
else:
    print("Both models have the same accuracy, indicating similar performance in this scenario.")