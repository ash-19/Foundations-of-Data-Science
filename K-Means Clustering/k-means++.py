import numpy as np
import scipy as sp
from numpy import genfromtxt
from scipy import linalg as LA
from sklearn.cluster import KMeans

Xp = np.array([[0,4], [3,0],[4, 18], [4, 12], [5, 15]])

print("Set X: \n", Xp)
print("==============================================================")
print("Standard Lloyd's run for 300 iterations:")
print("==============================================================")
kmeansP = KMeans(n_init = 1, n_clusters=3, random_state=0).fit(Xp)
print("Labels of each point: ", kmeansP.labels_)
print("Cluster centers (set S): \n", kmeansP.cluster_centers_)
print("----------------------------------------")
print("Cost(X,S): ", kmeansP.inertia_, "\n")
print("==============================================================")

print("k-means++ selection with the algo running for 1000 iterations:")
print("==============================================================")
kmeansP = KMeans(n_clusters=3, random_state=0, init='k-means++', max_iter=1000).fit(Xp)
print("Labels of each point: ", kmeansP.labels_)
print("Cluster centers (set S'): \n", kmeansP.cluster_centers_)
print("----------------------------------------")
print("Cost(X,S'): ", kmeansP.inertia_)
print("----------------------------------------")