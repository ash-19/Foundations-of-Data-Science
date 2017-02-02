import numpy as np
import scipy as sp
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from numpy import linalg as LA
from numpy import genfromtxt

# import the data from the .csv file to a 100x4 array. Randomize its order
data = genfromtxt('D4.csv', delimiter=',')
randomData = np.array(data)
np.random.shuffle(randomData)

# Select training data of size 70 (first 70 rows in randomized dataset).
q1TrainArray = randomData[0:70, [0,1,2,3]]
# Select testing data of size 30 (following rest of the 30 rows in randomized dataset)
q1TestArray = randomData[70:100, [0,1,2,3]]
print ("==================================================================")

def runAlgo():
    print ("Length of training data: " , len(q1TrainArray))
    print ("Length of testing data : " , len(q1TestArray))
    # Make xTrain[x1,x2,x3], yTrain[column4]
    xTrain = q1TrainArray[:, [0,1,2]]
    yTrain = q1TrainArray[:, [3]]

    # Make xTest[x1,x2,x3], yTest[column4]
    xTest = q1TestArray[:, [0,1,2]]
    yTest = q1TestArray[:, [3]]

    # solve for a line l which minimizes the SSE(P,l). This line equation is the model. 
    # Build the model from the training data.
    clf = linear_model.LinearRegression()
    clfFit = clf.fit(xTrain, yTrain)
    coefficents = clfFit.coef_[0]
    coefficents = np.array(coefficents)

    # Predicts the value of y using the found Model for the tuple of x values (x1,x2,x3, ...)
    # y = a0 + a1x1+ a2x2 + ...
    def predictY(xTuple):
      return (clfFit.intercept_ + (coefficents[0] * xTuple[0]) + (coefficents[1] * xTuple[1]) + (coefficents[2] * xTuple[2]))

    # Finds the SSE
    def sse():
          sumE = 0
          for i in range(0,len(xTest)):
                yHat = predictY(xTest[i])[0]
                #print (yHat)
                r = (yTest[i] - yHat)**2
                #print("r:" , r)
                sumE = sumE + r
          return sumE

    print ("Coefficients [a1, ..., an]: ", clfFit.coef_[0])
    print ("Intercept (a0)            : ", clfFit.intercept_)
    print ("SSE                       : ", sse())
    print ("==================================================================")
    
runAlgo()

# Select training vs testing data (90,10)
q1TrainArray = randomData[0:90, [0,1,2,3]]
q1TestArray = randomData[90:100, [0,1,2,3]]
runAlgo()

# Select training vs testing data (67,33)
q1TrainArray = randomData[0:67, [0,1,2,3]]
q1TestArray = randomData[67:100, [0,1,2,3]]
runAlgo()

# Select training vs testing data (80,20)
q1TrainArray = randomData[0:80, [0,1,2,3]]
q1TestArray = randomData[80:100, [0,1,2,3]]
runAlgo()
