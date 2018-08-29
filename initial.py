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

unique=getUniqueWords(unique)
unique_np=np.hstack(unique)
unique_np = np.unique(unique_np)

weight = [[0 for x in range(8960)] for y in range(50)] 

count = 0;

for z in range(len(unique)):
    print(z)
    nn = np.array(unique[z])
    for x in range(np.size(nn)):
        for y in range(np.size(unique_np)):
            if nn[x] == unique_np[y]:
                count = y
                break;
        weight[z][count] = weight[z][count] + 1        

#for z in range(len(unique)):
#    print(z)
#    nn = np.array(unique[z])
#    for x in range(np.size(unique_np)):
#        for y in range(np.size(nn)):
#            if unique_np[x] == nn[y]:
#                weight[z][x] = weight[z][x] + 1    
    

