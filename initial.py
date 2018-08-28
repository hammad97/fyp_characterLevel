#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 16:02:17 2018

@author: hammad
"""
import os
import re
import numpy as np
#from sklearn.cluster import KMeans

def getUniqueWords(allWords) :
    uniqueWords = [] 
    for i in allWords:
        if not i in uniqueWords:
            uniqueWords.append(i)
    return uniqueWords
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

unique=getUniqueWords(unique)
#myset= list(set(unique))
#print(unique)
unique_np=np.hstack(unique)




#unique_np= unique_np.flatten()
#print(unique_np[50])
print(unique_np)
print("Type"+str(unique_np.shape))
unique_np= np.unique(unique_np)
print(unique_np)
print("Type"+str(unique_np.shape))

#print(filedata)
 
#   works on float only       
#kmeans = KMeans(n_clusters=5)
#kmeans = kmeans.fit(filedata)
#labels = kmeans.predict(filedata)
#centroids = kmeans.cluster_centers_
#print(centroids)

