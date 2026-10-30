# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 20:56:42 2026

@author: VICTUS
"""

import pandas as pd

data = {
    'Study_Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Scores': [15, 25, 35, 45, 55, 65, 75, 85, 90, 95]
}

df = pd.DataFrame(data)

print(df)

#Data Analysis

print("Dataset Information:")
print(df.info())

print("\nFirst 5 Records:")
print(df.head())

print("\nStatistical Summary:")
print(df.describe())

print("\nAverage Score:")
print(df['Scores'].mean())

print("\nMaximum Score:")
print(df['Scores'].max())

print("\nMinimum Score:")
print(df['Scores'].min())

#Data Visualization

import matplotlib.pyplot as plt

plt.scatter(df['Study_Hours'], df['Scores'])

plt.xlabel('Study Hours')
plt.ylabel('Student Scores')
plt.title('Study Hours vs Student Performance')

plt.show()

#Machine Learning Model

from sklearn.linear_model import LinearRegression

X = df[['Study_Hours']]
y = df['Scores']

model = LinearRegression()
model.fit(X, y)

predicted_score = model.predict(pd.DataFrame({'Study_Hours': [6]}))

print("Predicted score for 6 study hours:", predicted_score[0])

#Model Evaluation

from sklearn.metrics import mean_absolute_error, r2_score

y_pred = model.predict(X)

mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("Mean Absolute Error:", mae)
print("R2 Score:", r2)
 

