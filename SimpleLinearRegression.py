#importing libraries
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

#importing dataset
dataset = pd.read_csv('LinearRegression2.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values



#splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

#fitting simple linear regression to the training set 
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#predicting the test set result
Y_pred = regressor.predict(X_test)
X_pred = regressor.predict(X_train)

#visualising the training set result
plt.scatter(X_train, Y_train, color = 'green')
plt.plot(X_train, X_pred, color = 'yellow' )
plt.title('Test set graph')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#visualising the test set result
plt.scatter(X_test, Y_test, color = 'blue')
plt.plot(X_train, X_pred, color = 'pink')
plt.title('Test set graph')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
