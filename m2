import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn import datasets 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report 

# Load the Iris dataset from sklearn 
iris = datasets.load_iris() 
# Convert the data into a DataFrame 
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names) 
iris_df['target'] = iris.target 
 
iris_df.shape 


iris_df.columns


iris_df.info()


iris_df.describe() 

# EDA: Pairplot to visualize rela onships between features 
sns.pairplot(iris_df, hue='target', palette='viridis', diag_kind='hist') 
plt.suptitle("Pairplot of Iris Dataset", y=1.02) 
plt.show()


# Other Plots 
plt.figure(figsize=(12, 6)) 
 
# Boxplot 
plt.subplot(1, 2, 1) 
sns.boxplot(x='target', y='sepal length (cm)', data=iris_df) 
plt.title('Boxplot of Sepal Length by Target') 
 
# Violin Plot 
plt.subplot(1, 2, 2) 
sns.violinplot(x='target', y='sepal width (cm)', data=iris_df) 
plt.title('Violin Plot of Sepal Width by Target') 
 
plt.tight_layout() 
plt.show()


# Count Plot (Bar Plot) 
plt.figure(figsize=(8, 5)) 
sns.countplot(x='target', data=iris_df, palette='viridis') 
plt.title('Count of Samples by Target') 
plt.xlabel('Target (Species)') 
plt.ylabel('Count') 
plt.xticks(ticks=[0, 1, 2], labels=iris.target_names) 
plt.show() 



# Correla on Heatmap 
plt.figure(figsize=(8, 6)) 
sns.heatmap(iris_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=1, linecolor='white') 
plt.title('Correla on Heatmap of Iris Features') 
plt.show()


# Separate features (X) and target (y) from the DataFrame 
X = iris_df.drop('target', axis=1)  # Features 
y = iris_df['target']               # Target (labels) 
 
# Split the dataset into training and tes ng sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
 
# Standardize features by removing the mean and scaling to unit variance 
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train) 
X_test = scaler.transform(X_test) 
 
# Ini alize the Logis c Regression model 
model = LogisticRegression() 
 
# Train the model on the training data 
model.fit(X_train, y_train) 
 
# Predict on the test data 
y_pred = model.predict(X_test) 
 
# Calculate training and tes ng accuracy 
train_accuracy = accuracy_score(y_train, model.predict(X_train)) 
test_accuracy = accuracy_score(y_test, y_pred)

print("Training Accuracy:", train_accuracy) 
print("Testing Accuracy:", test_accuracy) 



# Create a confusion matrix 
conf_matrix = confusion_matrix(y_test, y_pred) 
print("\nConfusion Matrix:") 
print(conf_matrix) 



# Print classifica on report 
print("\nClassifica on Report:") 
print(classification_report(y_test, y_pred, target_names=iris.target_names))



# Print probabili es of classifica on for the first few samples in the test set 
print("\nProbabili es of Classifica on:") 
probabilities = model.predict_proba(X_test[:5]) 
for i, prob in enumerate(probabilities): 
    print(f"Sample {i+1}: {list(zip(iris.target_names, prob))}") 

    
