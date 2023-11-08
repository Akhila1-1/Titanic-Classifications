# Titanic-Classifications
# Titanic Survival Prediction

This repository contains a Python script for building a machine learning model to predict the survival of passengers on the Titanic. The code utilizes the Titanic dataset, which includes information about passengers, such as their socio-economic status (SES), age, gender, and other factors, to make predictions about whether a passenger survived the sinking of the Titanic.

## Overview

The code is written in Python and uses popular libraries such as Pandas, NumPy, Seaborn, and Scikit-Learn for data preprocessing and modeling. It employs logistic regression, a common machine learning algorithm for binary classification tasks, to predict passenger survival.

## Features

- Loads the Titanic dataset from a local CSV file.
- Handles missing values by imputing them with the mean value of their respective columns.
- Performs data preprocessing by one-hot encoding categorical features like gender (Sex) and socio-economic status (Pclass).
- Splits the dataset into training and testing sets.
- Trains a logistic regression model on the training data.
- Evaluates the model's accuracy and provides a classification report.

## Usage

To use this code, follow these steps:

1. Download the Titanic dataset (CSV file) and save it to your local machine.
2. Update the file path to the dataset in the code.
3. Run the script to build and evaluate the logistic regression model.

Feel free to modify and enhance the code as needed for your specific use case or data analysis.
