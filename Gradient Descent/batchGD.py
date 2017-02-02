import pandas as pd
import numpy as np

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

iterations = 500
alpha = 0.08

# Add a columns of 1s as intercept to X. This becomes the last columns
# corresponding to alpha0
X_df['intercept'] = 1

X = np.array(X_df)
# print(X)
y = np.array(y_df).flatten()
# print(y)
theta = np.array([0, 0, 0, 0])

# Computes the cost of fitting all the data points X against the 
# true y.
def cost_function(X, y, theta):
    m = len(y) 
    ## Calculate the cost 
    F = np.sum((X.dot(theta)-y)**2)/2/m
    return F

# Performs Batch GD using the learning rate alpha for 
# iterations number of times.
def gradient_descent(X, y, theta, alpha, iterations):
    # Contains the cost function for each iteration
    cost_history = [0] * iterations
    
    print("                     alpha1      alpha2      alpha3      alpha0")
    for iteration in range(iterations):
        # hypothesis = X.dot(theta)
        # loss = hypothesis-y
        # gradient = X.T.dot(loss)/m
        # theta = theta - alpha*gradient
        # cost = cost_function(X, y, theta)
        # cost_history[iteration] = cost
        

        ## If you really want to merge everything in one line:
        theta = theta - alpha * (X.T.dot(X.dot(theta)-y)/m)
        cost = cost_function(X, y, theta)
        cost_history[iteration] = cost
        print(iteration, ") alpha values  : ", theta, "  Cost function value : ", cost)

    # return theta, cost_history


# (b, c) = gradient_descent(X, y, theta, alpha, iterations)
gradient_descent(X, y, theta, alpha, iterations)

# print();
# print("Learning rate       : ", alpha)
# print();
