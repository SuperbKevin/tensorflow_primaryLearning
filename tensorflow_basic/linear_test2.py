import numpy as np
from numpy.linalg import inv
from numpy import dot
from numpy import mat

#y=2x
x = mat([1, 2, 3]).reshape(3, 1)
y = 2*x

#theta = (x' x)^-1 x' y

# theta = dot(dot(inv(dot(x.T, x)), x.T), y)

#theta = theta - alpha*(theta*x-y)* x
theta = 1.
alpha = 0.1
for i in range(100):
    # theta = theta + np.sum(alpha * (y - dot(x, theta)) * x.reshape(1,3))/3
    theta = theta - alpha * dot(x.T, (dot(x, theta) - y))
print(theta)