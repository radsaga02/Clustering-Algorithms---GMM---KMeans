#Implementation of Scikit Learn GMM algorithm:
# The following code is writte in Python 3
# Group members: 
# 1) Aayush Sinha
# 2) Abhishek Jangalwa
# 3) Radhika Agarwal
from sklearn import mixture
import sys
import pandas as pd


#data = data=pd.read_csv(r"C:\Users\amiya\Desktop\USC GRAD\Machine Learning\Assignment\HW2\clusters.txt")
data=pd.read_csv(sys.argv[1], header=None)
points=mixture.GaussianMixture(n_components=3,covariance_type='full')
points.fit(data)
print ("Amplitudes :\n\n",points.weights_)
print ("\nMeans: \n\n",points.means_)
print ("\nCovariances: \n\n",points.covariances_)

