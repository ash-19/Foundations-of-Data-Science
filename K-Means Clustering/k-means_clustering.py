import numpy as np
import scipy as sp
from numpy import genfromtxt
from scipy import linalg as LA
from sklearn.cluster import KMeans

dataP = genfromtxt('P.csv', delimiter=',') 
dataQ = genfromtxt('Q.csv', delimiter=',')

Xp = np.array(dataP)
Xq = np.array(dataQ)

for i in range(1,11):
    kmeansP = KMeans(n_clusters=i, random_state=0, init='k-means++').fit(Xp)    
    print("%2d. " %(i), kmeansP.inertia_)

print("k anywhere between 5 and 10 is good for dataset P\n")

for i in range(1,11):
    kmeansQ = KMeans(n_clusters=i, random_state=0, init='k-means++').fit(Xq)    
    print("%2d. " %(i), kmeansQ.inertia_)

print("k anywhere between 6 and 10 is good for dataset Q\n")
