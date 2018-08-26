#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 16:02:17 2018

@author: hammad
"""
import os
#from sklearn.cluster import KMeans

filedata = []

for dirpath, dirnames, filenames in os.walk('Doc50'):
    print(filenames[4])
    
i=0

for file in filenames:
    fpath = os.getcwd()
    fpath = os.path.join(fpath, str(dirpath))
    fpath = os.path.join(fpath, str(filenames[i]))
    filee = open(fpath,mode='r')
    filedata.append(filee.read())
    filee.close()
    i=i+1

#print(filedata)
 
#   works on float only       
#kmeans = KMeans(n_clusters=5)
#kmeans = kmeans.fit(filedata)
#labels = kmeans.predict(filedata)
#centroids = kmeans.cluster_centers_
#print(centroids)

