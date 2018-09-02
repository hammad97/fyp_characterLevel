#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 16:02:17 2018

@author: hammad
"""
#
#from sklearn.decomposition import TruncatedSVD
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_extraction.text import HashingVectorizer
#from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.pipeline import make_pipeline
#from sklearn.preprocessing import Normalizer
#from sklearn import metrics
#
import time
import os
import math
import re
import numpy as np
#import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

start_time=  time.time()                                                        #Calculating initial time
filedata = []   


for dirpath, dirnames, filenames in os.walk('Doc50'):                           #Reading all files in Doc50 folder and storing it names in
    print(filenames[4])                                                         #filenames and path in dirpath
    
i=0


for file in filenames:
    filedata2= ''
    fpath = os.getcwd()
    fpath = os.path.join(fpath, str(dirpath))
    fpath = os.path.join(fpath, str(filenames[i]))                              #fpath now contains whole path to each file at each iteration
    filee = open(fpath,mode='r')
    filedata2=filee.read()                                                      #here file is read and stored in a temporary variable
    filedata2 = str.lower(filedata2)                                            
    filedata2=''.join(e for e in filedata2 if e.isalpha() or e==' ')            #here file is normalized from noisy text data by only keeping alphabetical values
    filedata2=re.sub(' +',' ',filedata2)                                        #after performing above step there is a increase in spaces to remove them this step is done
    filedata.append(filedata2)                                                  #finally file data is moved to actual variable from temporary one
    filee.close()
    i=i+1

unique = []

for fdata in filedata:                                                          #each files data was stored in one single string therefore each string is split ...
    neww=fdata.split(' ')                                                       #...to obtain all the words from that document to be later used in VSM
    neww=neww[:-1]                                                              #each document's last index contains ' ' as a feature which is removed from every where
    unique.append(neww)                                                         #IMPORTANT NOTE: unique doesnot contain unique words of all files it is just the variable name...                    
                                                                                #...containing 2d list where each row contains list of words used in one file

unique_np=np.hstack(unique)                                                     

unique_np= np.unique(unique_np)                                                 #unique_np now is the dictionary containing all unique words from all 50 docs
unique_npp=np.array(unique_np)

uniq2= []
for x in unique_npp:
    if len(x)>2:
        uniq2.append(x)
unique_np=np.array(uniq2)                                                       #now unique_mp contains unique words too but this time only of length >2

weights = [[0 for x in range(unique_np.size)] for y in range(50)]               #creation of a 2d list for holding weights for each word with respect to its doc

k=0
l=0
for fdata2 in unique:
    l=0
    for x in unique_np:
        weights[k][l]=fdata2.count(x)
        l=l+1
    k=k+1                                                                       #now we have tf stored in weights with respect to document number
    
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
km = KMeans(n_clusters=5, init='k-means++', max_iter=100, n_init=1,verbose=0)

print("Clustering sparse data with %s" % km)

km.fit(weights_npp)

print()

print("Labels:" + str(km.labels_))

print()

pred_classes = km.predict(weights_npp)

filenames_np = np.array(filenames)

for cluster in range(5):
    print("Cluster: ", cluster)
    print(filenames_np[np.where(pred_classes == cluster)])

#print("Top terms per cluster:")
#
#order_centroids = km.cluster_centers_.argsort()[:, ::-1]
#
#terms = unique_np
#for i in range(5):
#    print("Cluster %d:" % i, end='')
#    for ind in order_centroids[i, :10]:
#        print(' %s' % terms[ind], end='')
#    print()




    
print("Execution time:"+ str(time.time()-start_time))
