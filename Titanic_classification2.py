#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

# Load the Titanic dataset from a local file
titanic = pd.read_csv("C:\\Users\\maheh\\Downloads\\tested.csv")

# Impute missing values with the mean value of the respective column
imputer = SimpleImputer(strategy="mean")
titanic[["Age", "Fare"]] = imputer.fit_transform(titanic[["Age", "Fare"]])

# Data preprocessing
X = titanic[["Pclass", "Sex", "Age", "SibSp", "Fare"]]
X = pd.get_dummies(X, columns=["Sex"], drop_first=True)
X = pd.get_dummies(X, columns=["Pclass"], drop_first=True)
y = titanic["Survived"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[ ]:




