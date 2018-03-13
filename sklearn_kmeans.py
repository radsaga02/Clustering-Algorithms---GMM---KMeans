#Implementation of Sklearn Kmeans algorithm
# The following code is writte in Python 3
# Group members: 
# 1) Aayush Sinha
# 2) Abhishek Jangalwa
# 3) Radhika Agarwal
from sklearn import cluster
import sys
import pandas as pd

#data=pd.read_csv(r"C:\Users\amiya\Desktop\USC GRAD\Machine Learning\Assignment\HW2\clusters.txt")
data=pd.read_csv(sys.argv[1], header=None)
points=cluster.KMeans(n_clusters=3)
points.fit(data)
print ("resultant centroids are: \n\n",points.cluster_centers_)


    
