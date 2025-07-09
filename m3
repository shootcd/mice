###Implements Multinomial Logistic Regression (Iris Dataset)

import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
 
from sklearn.datasets import load_iris 
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, classification_report 
 
# Load Iris data 
iris = load_iris() 
df = pd.DataFrame(data=iris.data, columns=iris.feature_names) 
df['species'] = iris.target 
df['species_name'] = df['species'].map(dict(zip(range(3), iris.target_names))) 
 
df.head()



print(df.info()) 
print(df.describe()) 
 
# Check class distribu on 
print(df['species_name'].value_counts())



plt.figure(figsize=(6,4)) 
sns.countplot(x='species_name', data=df) 
plt.title("Count of Each Iris Species") 
plt.xlabel("Species") 
plt.ylabel("Count") 
plt.show()


plt.figure(figsize=(8,6)) 
sns.heatmap(df.iloc[:, :4].corr(), annot=True, cmap="coolwarm", fmt=".2f") 
plt.title("Correla on Heatmap of Iris Features") 
plt.show()

sns.pairplot(df, hue='species_name', vars=iris.feature_names[:3]) 
plt.suptitle("Pairplot of Selected Features", y=1.02) 
plt.show() 




X = df[iris.feature_names] 
y = df['species'] 
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) 
 
# Use mul nomial for mul-class classifica on with somax 
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200) 
model.fit(X_train, y_train) 
 
# Predict 
y_pred = model.predict(X_test) 
 
print("Classifica on Report:") 
print(classification_report(y_test, y_pred, target_names=iris.target_names)) 




cm = confusion_matrix(y_test, y_pred) 
plt.figure(figsize=(6,4)) 
sns.heatmap(cm, annot=True, cmap="YlGnBu", fmt='d', 
            xticklabels=iris.target_names, yticklabels=iris.target_names) 
plt.xlabel('Predicted') 
plt.ylabel('Actual') 
plt.title("Confusion Matrix Heatmap") 
plt.show()



print("Training Accuracy:", model.score(X_train, y_train)) 
print("Test Accuracy:", model.score(X_test, y_test))


