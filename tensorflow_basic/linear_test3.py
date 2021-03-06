import numpy as np
from numpy.linalg import inv
from numpy import dot
from numpy import mat
import pandas as pd

data_set = pd.read_csv('data.csv')
# print(data_set)

temp = data_set.iloc[:, 2:5]
temp['x0'] = 1
X = temp.iloc[:, [3, 0, 1, 2]]
# print(X)
Y = data_set.iloc[:, 1].reshape(150, 1)
# print(Y)
theta = dot(dot(inv(dot(X.T,X)), X.T), Y)
print(theta.reshape(4, 1))

theta = np.array([1.,1.,1.,1.]).reshape(4,1)
alpha = 0.1
temp = theta
X0 = X.iloc[:, 0].reshape(150, 1)
X1 = X.iloc[:, 1].reshape(150, 1)
X2 = X.iloc[:, 2].reshape(150, 1)
X3 = X.iloc[:, 3].reshape(150, 1)

for i in range(10000):
    temp[0] = theta[0] + alpha*np.sum((Y - dot(X,theta))*X0)/150
    temp[1] = theta[1] + alpha * np.sum((Y - dot(X, theta)) * X1) / 150
    temp[2] = theta[2] + alpha * np.sum((Y - dot(X, theta)) * X2) / 150
    temp[3] = theta[3] + alpha * np.sum((Y - dot(X, theta)) * X3) / 150
    theta = temp
print(theta)