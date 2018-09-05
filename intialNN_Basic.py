#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 22:30:10 2018

@author: hammad
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import os
import re


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
tfidf_vector = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vector.fit_transform(filedata)
print('tfidf_matrix shape: ',tfidf_matrix.shape)


points_n = 200
clusters_n = 5
iteration_n = 100

points = tf.constant(np.random.uniform(0, 10, (points_n, 2)))
centroids = tf.Variable(tf.slice(tf.random_shuffle(points), [0, 0], [clusters_n, -1]))

points_expanded = tf.expand_dims(points, 0)
centroids_expanded = tf.expand_dims(centroids, 1)

distances = tf.reduce_sum(tf.square(tf.subtract(points_expanded, centroids_expanded)), 2)
assignments = tf.argmin(distances, 0)

means = []
for c in range(clusters_n):
    means.append(tf.reduce_mean(
      tf.gather(points, 
                tf.reshape(
                  tf.where(
                    tf.equal(assignments, c)
                  ),[1,-1])
               ),reduction_indices=[1]))

new_centroids = tf.concat(means, 0)

update_centroids = tf.assign(centroids, new_centroids)
init = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init)
  for step in range(iteration_n):
    [_, centroid_values, points_values, assignment_values] = sess.run([update_centroids, centroids, points, assignments])
    
  print("centroids" + "\n", centroid_values)

plt.scatter(points_values[:, 0], points_values[:, 1], c=assignment_values, s=50, alpha=0.5)
plt.plot(centroid_values[:, 0], centroid_values[:, 1], 'kx', markersize=15)
plt.show()


print("Neural Execution time:"+ str(time.time()-start_time))