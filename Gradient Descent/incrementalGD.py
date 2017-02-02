import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('D4.csv', names = ['x1', 'x2', 'x3', 'y'])
# print(data.head())
# print(data)

# Split explanatory and dependent variables
X_df = pd.DataFrame(data, columns=['x1', 'x2', 'x3'])
y_df = pd.DataFrame(data.y)
# print(X_df)
# print(y_df)
# Number of data points
m = len(y_df)

# iterations = 10000
# alpha = 0.01 
iterations = 2099
alpha = 0.08

## Add a columns of 1s as intercept to X. This becomes the last columns
# corresponding to alpha0
X_df['intercept'] = 1

X = np.array(X_df)
# print(X)
y = np.array(y_df).flatten()
# print(y)
theta = np.array([0, 0, 0, 0])

# Computes the cost of fitting the ith data point against the true y_i.
def cost_function(X, y, theta):
    ## Calculate the cost with the given parameters
    F = ((X.dot(theta)-y)**2)/2
    return F

# cost_function(X, y, theta)
# print(cost_function(X, y, theta))
# print(theta)
# Performs Iterative GD using the learning rate alpha for 
# iterations number of times. Gradient is calculated on only single 
# data point at a time.
def gradient_descent(X, y, theta, alpha, iterations):
    
    i = 0
    cost_history = [0] * iterations
    
    print("                     alpha1      alpha2      alpha3      alpha0")
    for iteration in range(iterations):
        yCap = X[i].dot(theta)
        residue = yCap-y[i]
        gradient = X[i].dot(residue)
        theta = theta - alpha * gradient
        cost = cost_function(X[i], y[i], theta)
        cost_history[iteration] = cost
        i = (i+1) % m
        print(iteration, ") alpha values  : ", theta, "  Cost function value : ", cost)

    # return theta, cost_history


# (b, c) = gradient_descent(X, y, theta, alpha, iterations)
gradient_descent(X, y, theta, alpha, iterations)

# print();
# print("Learning rate       : ", alpha)
# print();
