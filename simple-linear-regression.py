import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# importing datasets
dataset = pd.read_csv('Salary_Data.csv')


# Here x is the independent variable (experience) and y is the depedent variable (salary)

# select all and take all the columns except the last one
x = dataset.iloc[:, :-1].values
# select all and take the index 1 column
y = dataset.iloc[:, 1].values


# Here test set is 80% and train set is 20%
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=4/5, random_state=0)


""" A simple linear regressor model will be created by using the train set (x, y)
 Now this model can predict any y value (here it's salary) on a given x value (years of experience)
 test set gives us visibility of how good this model is at predicting
"""


# Fitting simple linear regression to the training set
regressor = LinearRegression()
regressor.fit(x_train, y_train)


# This will predict the y value for given x value
# Now we can compare between y_test and y_predicted to see how well it's predicting
y_test_predicted = regressor.predict(x_test)
y_train_predict = regressor.predict(x_train)


# Visualizing the prediction line for the training set

# Creates a scatter graph of real data of train set
plt.scatter(x_train, y_train, color="red")
# Creates a plot of x and y
plt.plot(x_train, y_train_predict, color="blue")
plt.xlabel("experience")
plt.ylabel("salary")
plt.show()


# Visualizing the prediction line for the testing set

# Creates a scatter graph of real data of test set
plt.scatter(x_test, y_test, color="red")
plt.plot(x_train, y_train_predict, color="blue")
plt.xlabel("experience")
plt.ylabel("salary")
plt.show()

""" Here while visualizing the test set the plot looks incomplete
 becoz the train set doesnot provide information for a greater range
 To avoid this you can choose to plot with test set data or give more data to train set
"""
