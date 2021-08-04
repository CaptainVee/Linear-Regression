#importing libraries
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

#importing dataset
dataset = pd.read_csv('pheobe.csv')

X = dataset.iloc[:500, :-1].values
Y = dataset.iloc[:500, 3].values

#splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/4, random_state = 0)

#fitting simple linear regression to the training set 
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#predicting the test set result
Y_pred = regressor.predict(X_test)

#visualising the result
plt.plot(Y_pred, color = 'red', label = 'pred')
plt.plot(Y_test, color = 'blue', label = 'actual')
plt.title('Ovulation Prediction')
plt.xlabel('Time')
plt.ylabel('Ovulation')
plt.legend()
plt.show()

