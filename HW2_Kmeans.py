#Implementation of Kmeans algorithm
# The following code is writte in Python 3
# Group members: 
# 1) Aayush Sinha
# 2) Abhishek Jangalwa
# 3) Radhika Agarwal
import numpy as np
import pandas as pd
import math
import sys

iteration=0
max_iteration=50
precision=0.0001
final_centroids=[]
def diffe(a,b):
    difference=[]
    for i in range(len(a)):
        d=((a[i]-b[i])**2)
        difference.append(d)
        res=math.sqrt(sum(difference))
    return res
def K_Means(no_of_centroids,df,centroids):
    diff=[]
    global iteration
    global max_iteration
    global final_centroids
    iteration+=1
    if len(centroids)<=0:
        for k in range(no_of_centroids): #creating random centroids for the first time
            rand_cen=(df.iloc[np.random.choice(len(df)-1)]).tolist()
            if rand_cen in centroids:
                k-=1
                #print ("Duplicate Centroid encountered!")
                sys.exit()
                continue
            else:
                centroids.append(rand_cen)
             
        #print("\nRandomly selected centroids are: ",centroids)
    #else:
    #    print("\n====\nCentroids:\n",centroids)    
    
    for k in range(no_of_centroids):
        b=[]
        for i in range(len(df)):
            res=diffe(centroids[k],(df.x[i],df.y[i]))
            b.append(res)
        diff.append(b)
    values=[]
    for j in range(len(df)):
        anything=[]
        for i in range(len(diff)):
            anything.append(diff[i][j])
        values.append(anything)    
    #print("\nList of the differences of each value in main df from each centroid: ",values)
    
    indices=[]
    for distances in values:
        m=min(distances)
        i=distances.index(m)
        indices.append(i)
    
    temp_group={}
    for ki in range(no_of_centroids):
        temp_group[ki]=[]
    for i in range(len(df)):
        temp_group[indices[i]].append(df.iloc[i])
    avg=[]
    for ki in range(no_of_centroids):
        ji_x,ji_y=0,0
        len_temp=temp_group[ki]
        for ji in range(len(len_temp)):
            ji_x+=np.mean(temp_group[ki][ji][0])
            ji_y+=np.mean(temp_group[ki][ji][1])
        avg.append(ji_x/len(len_temp))
        avg.append(ji_y/len(len_temp))
    new_centroids=[]
    for i in range(0,len(avg),2):
        new_centroids.append(avg[i:i+2])
    
    flag=1
    for i in range(len(centroids)):
        if abs(centroids[i][0]-new_centroids[i][0])>precision: flag=-1
        if abs(centroids[i][1]-new_centroids[i][1])>precision: flag=-1
    
    if flag!=1:
        #print("\nCentroids changed! Iteration:",iteration,"\n")
        if(iteration>=max_iteration):
            #print("\n========\nMax iteration reached:",iteration)
            return (new_centroids)
        else: 
            return K_Means(no_of_centroids,df,new_centroids)
    else:
        #print("\n========\nOptimal solution found after ",iteration," iterations: ")
        return(new_centroids)

def main():
    try:
        #data=pd.read_csv(r"C:\Users\amiya\Desktop\USC GRAD\Machine Learning\Assignment\HW2\clusters.txt", header = None)
        data=pd.read_csv(sys.argv[1], header=None)
        df=pd.DataFrame(data)
        df.columns=['x','y']
        no_of_centroids=3
        final_centroids=K_Means(no_of_centroids,df,[])
        print(final_centroids)
    except Exception as e:
        print(e)
        print("Syntax: python <program_name.py> <data_file.txt>")
        exit()
        
if __name__ == "__main__":
    main()            
