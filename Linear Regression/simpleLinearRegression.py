import numpy as np
import matplotlib as mpl
mpl.use('PDF')
import matplotlib.pyplot as plt
import scipy as sp
import math
from numpy import linalg as LA
from numpy import genfromtxt
from random import randint

def column(matrix, i):
    return [row[i] for row in matrix]

# import the data from the .csv file to a 100x4 array. Randomize its order
data = genfromtxt('HW03/D3.csv', delimiter=',')
randomData = np.array(data)
np.random.shuffle(randomData)

# Select training data of size 70 (first 70 rows in randomized dataset),
# Select testing data of size 30 (following rest of the 30 rows in randomized dataset)
q1TrainArray = randomData[0:70, [0,3]]
q1TestArray = randomData[70:100, [0,3]]
print ("Length of training data: ", len(q1TrainArray))
print ("Length of testing data : ", len(q1TestArray))

# Make xTrain[column1], yTrain[column4]
xTrain = column(q1TrainArray, 0)
yTrain = column(q1TrainArray, 1)

# Make xTest[column1], yTest[column4]
xTest = column(q1TestArray, 0)
yTest = column(q1TestArray, 1)


def plot_poly(xT,yT,xE,yE,p, split):
  plt.scatter(xT,yT, s=20, c="blue")
  plt.scatter(xE,yE, s=20, c="yellow")
  plt.axis([0,4,-6,10])
  s=sp.linspace(0,10,101)
  
  # solve for a line l which minimizes the SSE(P,l). This line equation is the model. 
  # Build the model from the training data.
  coefs=sp.polyfit(xT,yT,p)	# finds the least square polynomial fit and returns the coeff. (find the real model here)
  #print coefs
  ffit = np.poly1d(coefs)	# builds back the poly equation using the coeff vector. This is the y^ = M(x) equation
  print (ffit)
  
  # plot the line, for every x, compute y from line equation ffit and plot (x,y). Total 100 pts.
  plt.plot(s,ffit(s),'r-',linewidth=2.0)
  
  #evaluate the test data xE on the model made using the training data. 
  resid = ffit(xE)			# resid is y^ = M(x)
  # Use test data vector yE and the predicting values vector (y^) to find SSE of of the Model's prediction
  RMSE = LA.norm(resid-yE)		# y-M(x) (y - y^ for all y in testing data)
  SSE = RMSE * RMSE		

  print ("x = 1: y^: ", str((ffit(1))))
  print ("x = 2: y^: ", str((ffit(2))))
  print ("x = 3: y^: ", str((ffit(3))))
  print ("SSE      : ", SSE)

  # Plot the graph 
  title = "(%s, %s) split | degree %s fit | SSE %0.3f" % (split[0], split[1], p, SSE)
  plt.title(title)
  file = "HW3-Q1-(%s, %s)Fit-degree%s.pdf" % (split[0], split[1], p)
  plt.savefig(file, bbox_inches='tight')
  plt.clf()
  plt.cla()

# Different poly models
p_vals = [1,2,3,4,5]

def runAlgo(sp) :
      for i in p_vals:
        print ("--------------------------")
        print ("Degree p = ", str(i))
        print ("--------------------------")
        plot_poly(xTrain,yTrain,xTest,yTest,i, sp)

print ("*************************************************")
runAlgo([70,30])

print ("*************************************************")
# Select (training data, testing) data set (90,10)
q1TrainArray = randomData[0:90, [0,3]]
q1TestArray = randomData[90:100, [0,3]]
# Make xTrain[column1], yTrain[column4]
xTrain = column(q1TrainArray, 0)
yTrain = column(q1TrainArray, 1)
# Make xTest[column1], yTest[column4]
xTest = column(q1TestArray, 0)
yTest = column(q1TestArray, 1)
print ("Length of training data: ", len(q1TrainArray))
print ("Length of testing data : ", len(q1TestArray))
runAlgo([90,10])

print ("*************************************************")
# Select (training data, testing) data set (80,20)
q1TrainArray = randomData[0:80, [0,3]]
q1TestArray = randomData[80:100, [0,3]]
# Make xTrain[column1], yTrain[column4]
xTrain = column(q1TrainArray, 0)
yTrain = column(q1TrainArray, 1)
# Make xTest[column1], yTest[column4]
xTest = column(q1TestArray, 0)
yTest = column(q1TestArray, 1)
print ("Length of training data: ", len(q1TrainArray))
print ("Length of testing data : ", len(q1TestArray))
runAlgo([80,20])

print ("*************************************************")
# Select (training data, testing) data set (67,33)
q1TrainArray = randomData[0:67, [0,3]]
q1TestArray = randomData[67:100, [0,3]]
# Make xTrain[column1], yTrain[column4]
xTrain = column(q1TrainArray, 0)
yTrain = column(q1TrainArray, 1)
# Make xTest[column1], yTest[column4]
xTest = column(q1TestArray, 0)
yTest = column(q1TestArray, 1)
print ("Length of training data: ", len(q1TrainArray))
print ("Length of testing data : ", len(q1TestArray))
runAlgo([67,33])
