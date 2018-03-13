#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 00:19:15 2017

@author: Rads
"""

import numpy as np
import pandas as pd
import sys

class GMM:
    
    def __init__(self, clusters , iteration, threshold = 6 ,mean = [], cov = [], amplitude = [], conv = 0):
        self.clusters = clusters
        self.iteration = iteration
        self.threshold = threshold
        self.mean = mean
        self.cov = cov
        self.amplitude = amplitude
        self.conv = conv

    def fit(self, data):
        iter = 1
        weights = self.assign(data)
        conv = self.conv
        while True:
            self.mStep(data, weights)
            new_weights = self.eStep(data)
            if conv != 0 and self.conv != 0:
                if(np.abs(conv - self.conv) < self.threshold):
                    break
            if iter >= self.iteration:
                break
            weights = new_weights
            conv = self.conv
            iter += 1
        return
        
    def assign(self, data):
        clusters = self.clusters
        weights = []
        for datapoints in range(len(data)):
             pdf = np.random.random(clusters)
             total = float(sum(pdf))
             pdf = [i/total for i in pdf] 
             weights.append(pdf)       
        return weights
            
    def eStep(self, data):
        data = data.values
        length = len(data)
        weights = []
        mean = self.mean
        cov = self.cov
        amp = self.amplitude
        clusters = self.clusters
        conv = []
        for i in range(length):
            pdf = []
            for n in range(clusters):
                invCov = np.linalg.inv(cov[n])
                divFact = 1/np.sqrt((2 * np.pi ** 2) * np.linalg.det(cov[n]))
                tmpMatrix = data[i] - mean[n]
                tmpMatrix = np.matrix(tmpMatrix)
                transMatrix = tmpMatrix.T
                fact = np.dot(-0.5 * tmpMatrix, invCov)
                fact = np.dot(fact, transMatrix)
                fact = divFact * np.exp(fact)
                fact = fact * amp[n]
                fact = np.asscalar(fact)
                pdf.append(fact)
            total = float(sum(pdf))
            pdf  = [i/total for i in pdf] 
            pdf = np.asarray(pdf)
            weights.append(pdf)
            total = np.log(total)
            conv.append(total)
        self.conv = sum(conv)
        return weights
                    
                
    def mStep(self,data,weights=[]):
        tempData = []
        mean = []
        diffMatrix = []
        covariance = []
        amplitude = []
        clusters = self.clusters
        totalWeight = []
        for n in range(clusters):
            tmpWeight = 0
            for i in range(len(weights)):
                tmpWeight +=  weights[i][n]
                tempData.append(data.iloc[i] * weights[i][n])
            totalWeight.append(tmpWeight)    
            mean.append(np.sum(tempData, axis = 0))
            mean[n] = mean[n]/totalWeight[n]  
            tmpAmp = float("{0:.3}".format(totalWeight[n]/len(data)))
            amplitude.append(tmpAmp)
            tmpMatrix = [data.iloc[j] - mean[n] for j in range(len(data))]
            tmpMatrix = np.matrix(tmpMatrix)
            diffMatrix.append(tmpMatrix)
            tmpCov = [weights[k][n] * (np.outer(diffMatrix[n][k], np.transpose(diffMatrix[n][k]))) for k in range(len(data)) ]
            covariance.append(sum(tmpCov)/ totalWeight[n])
            
        self.mean = mean
        self.cov = covariance
        self.amplitude = amplitude
        return

def main():
    
    trainFile = sys.argv[1]

    #data=pd.read_csv(r"/Users/Rads/Downloads/INF 552 - Machine Learning/hw2/clusters.txt", header = None)
    
    data = pd.read_csv(trainFile, header = None)
    
    no_of_clusters=3
    no_of_iteration = 100
    
    output = GMM(no_of_clusters,no_of_iteration)
    output.fit(data)

    for i in range(no_of_clusters):
        print "Mean of the cluster ", i," is: ", output.mean[i].tolist()
        print "Amplitude of the cluster ", i," is: ",output.amplitude[i]
        print "Covariance matrix of the cluster ",  i," is: ", output.cov[i], "\n"

    
if __name__ == '__main__':
    main()






