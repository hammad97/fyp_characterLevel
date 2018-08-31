#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 16:02:17 2018

@author: hammad
"""
import time
import os
import math
import re
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

def getUniqueWords(allWords) :
    uniqueWords = [] 
    for i in allWords:
        if not i in uniqueWords:
            uniqueWords.append(i)
    return uniqueWords

start_time=  time.time()
filedata = []


for dirpath, dirnames, filenames in os.walk('Doc50'):
    print(filenames[4])
    
i=0


for file in filenames:
    filedata2= ''
    fpath = os.getcwd()
    fpath = os.path.join(fpath, str(dirpath))
    fpath = os.path.join(fpath, str(filenames[i]))
    filee = open(fpath,mode='r')
    filedata2=filee.read()
    filedata2 = str.lower(filedata2)
    filedata2=''.join(e for e in filedata2 if e.isalpha() or e==' ')
    filedata2=re.sub(' +',' ',filedata2)
    filedata.append(filedata2)
    filee.close()
    i=i+1

unique = []

for fdata in filedata:
    neww=fdata.split(' ')
    neww=neww[:-1]
    unique.append(neww)
#    unique= unique[:-1]    
unique2 = list(unique)
unique=getUniqueWords(unique)
#myset= list(set(unique))
#print(unique)
unique_np=np.hstack(unique)

#np.delete(unique_np)
#np.empty(unique_np)



#unique_np= unique_np.flatten()
#print(unique_np[50])
#print(unique_np)
#print("Type"+str(unique_np.shape))

unique_np= np.unique(unique_np)
unique_npp=np.array(unique_np)

uniq2= []
for x in unique_npp:
    if len(x)>2:
        uniq2.append(x)
unique_np=np.array(uniq2)
#print(unique_np)
#print("Type"+str(unique_np.shape))

#weights = []

weights = [[0 for x in range(unique_np.size)] for y in range(50)]

k=0
l=0
for fdata2 in unique:
    l=0
    for x in unique_np:
        weights[k][l]=fdata2.count(x)
        l=l+1
    k=k+1
    
"""
i=0
count=0

    
for fdata2 in unique:
    nn=fdata2[i:]
    nn=np.array(nn)
    for x in unique_np:
        for y in nn:
            if x == y:
                weights[i][count]=weights[i][count]+1       
        count=count+1
    i=i+1
    count=0
"""
#idf = [sum(x) for x in zip(*weights)]

idf = [0 for x in range(unique_np.size)]
i=0
for i in range(unique_np.size): #rows
    for j in range(50):
        if weights[j][i]>0:
            idf[i]=idf[i]+1
    


j=0
for i in idf:
    idf[j]=math.log10(50/i)
    j=j+1

weights_np = np.array(weights)
idf_np = np.array(idf)
weights_npp = weights_np * idf_np

#   works on float only       
kmeans = KMeans(n_clusters=5)
kmeans = kmeans.fit(weights_npp)
labels = kmeans.predict(weights_npp)
centroids = kmeans.cluster_centers_
print(centroids)
plt.scatter(centroids[0],centroids[1])
plt.show()

    
print("Execution time:"+ str(time.time()-start_time))
#    for index, x in np.ndenumerate(unique_np):
##        print(x)
#        for index2,y in np.ndenumerate(nn):
##            print(y)
#        #print(str(index)+" "+str(x))
#            if  x == y:
##                print("")
#                weights[i][count]=weights[i][count]+1       
#        count=count+1
#    i=i+1
#    count=0       

    
    
    
#    nn=fdata2[i:]
#    nn=np.array(nn)
    

#    for j in np.nditer(nn):
#        for count in unique_np:
#            if  np.equal(unique_np[count] , nn[j]):
#                weights[i,count]=weights[i,count]+1       
#    i=i+1       
    
        
        

#print(filedata)
 
#   works on float only       
#kmeans = KMeans(n_clusters=5)
#kmeans = kmeans.fit(filedata)
#labels = kmeans.predict(filedata)
#centroids = kmeans.cluster_centers_
#print(centroids)

