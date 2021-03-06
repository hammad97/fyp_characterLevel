#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 19:56:08 2018

@author: hammad
"""

from nltk.stem.snowball import EnglishStemmer

from nltk.tokenize import word_tokenize

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

import time
import os
import math
import re
import numpy as np
import pandas as pd
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
filedata3=[]
filedata4=[]
                                                    #...containing 2d list where each row contains list of words used in one file
ps=PorterStemmer()
for w in filedata:
    filedata3.append(ps.stem(w))
   
unique = []

for fdata in filedata3:                                                          #each files data was stored in one single string therefore each string is split ...
    neww=fdata.split(' ')                                                       #...to obtain all the words from that document to be later used in VSM
    neww=neww[:-1]                                                              #each document's last index contains ' ' as a feature which is removed from every where
    unique.append(neww)                                                         #IMPORTANT NOTE: unique doesnot contain unique words of all files it is just the variable name...                    
                        
count_vec = CountVectorizer(stop_words='english',
                            ngram_range=(1, 1), max_df=0.2, min_df=0.1, max_features=None)
#count_train = count_vec.fit(filedata)
#bag_of_words = count_vec.transform(filedata)
bag_of_words2=count_vec.fit_transform(filedata)

#print('dsdsdsds',count_vec.get_feature_names())
#tfidfmatrix= count_vec.it_transform(filedata)
tfidf_vector = TfidfVectorizer( sublinear_tf = True , max_df= 0.8 , min_df = 0.1,stop_words="english")
#words= word_tokenize(tfidf_vector)
#print(tfidf_vector)

    
tfidf_matrix = tfidf_vector.fit_transform(filedata)
print(tfidf_matrix)
print('tfidf_matrix shape: ',tfidf_matrix.shape)


num_clusters = 5

km = KMeans(n_clusters=num_clusters, init='k-means++')
km.fit(tfidf_matrix)
pred_classes = km.predict(tfidf_matrix)
filenames_np = np.array(filenames)

for cluster in range(num_clusters):
    print("Cluster: ", cluster)
    print(filenames_np[np.where(pred_classes == cluster)])
    

print("Execution time:"+ str(time.time()-start_time))
