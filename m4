###Implement SVM classifier (Iris Dataset) 


import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn import datasets 
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC 
from sklearn.metrics import classification_report, accuracy_score 

# Load the Iris dataset from sklearn 
iris = datasets.load_iris() 
# Convert the data into a DataFrame 
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names) 
iris_df['target'] = iris.target 


# Exploratory Data Analysis (EDA) 
print(iris_df.describe())  # Summary sta s cs 
print(iris_df.head())      # View first few rows


# Visualiza on (pairplot for all features) 
import seaborn as sns 
# EDA: Pairplot to visualize rela onships between features 
sns.pairplot(iris_df, hue='target', palette='viridis', diag_kind='hist') 
plt.suptitle("Pairplot of Iris Dataset", y=1.02) 
plt.show() 

# Separate features (X) and target (y) from the DataFrame 
X = iris_df.drop('target', axis=1)  # Features 
y = iris_df['target']               # Target (labels) 
 
# Split the dataset into training and tes ng sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
 
# Create the SVM model 
clf = SVC(kernel='linear')  # Experiment with different kernels (e.g., 'rbf') 
 
# Train the model 
clf.fit(X_train, y_train) 
 
# Make predic ons on the tes ng set 
y_pred = clf.predict(X_test) 
 
# Evaluate model performance 
print(classification_report(y_test, y_pred)) 
print("Training Accuracy:", accuracy_score(y_train, clf.predict(X_train))) 
print("Testing Accuracy:", accuracy_score(y_test, y_pred))


from sklearn.model_selection import GridSearchCV 
# Define a parameter grid to explore 
param_grid = {'kernel': ['linear', 'rbf'], 
              'C': [0.01, 0.1, 1, 10, 100]} 
# Create the GridSearchCV object 
grid_search = GridSearchCV(SVC(), param_grid, cv=5)  # 5-fold cross-valida on 
# Fit the grid search to the training data 
grid_search.fit(X_train, y_train) 
# Get the best model and its parameters 
best_model = grid_search.best_estimator_ 
best_params = grid_search.best_params_ 
print(best_params) 
# Use the best model for predic on and evalua on 
y_pred = best_model.predict(X_test) 
print(classification_report(y_test, y_pred)) 
print("Tes ng Accuracy:", accuracy_score(y_test, y_pred))


