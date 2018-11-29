#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 21:18:02 2018

@author: hammad
"""

import time
import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
import nltk
from collections import  Counter
from string import punctuation
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity



from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups()

text = ' '.join(data.data).lower()
print(text[100:350])

sentences_text = nltk.sent_tokenize(text)
print(len(sentences_text))

sentences = [nltk.word_tokenize(s) for s in sentences_text]
print(sentences[10])

min_count = 5
puncs = set(punctuation)
stops = set(stopwords.words('english'))

flat_words = []
for sentence in sentences:
    flat_words += sentence
    
counts = Counter(list(flat_words))
counts = pd.DataFrame(counts.most_common())
counts.columns = ['word', 'count']

counts = counts[counts['count'] >= min_count]
counts = counts[~counts['word'].isin(puncs)]
counts = counts[~counts['word'].isin(stops)]


vocab = pd.Series(range(len(counts)), index=counts['word']).sort_index()

print('The vocabulary has:', len(vocab), 'words')

filtered_sentences = []

for sentence in sentences:
    sentence = [word for word in sentence if word in vocab.index]
    if len(sentence):
        filtered_sentences.append(sentence)
sentences = filtered_sentences

for i, sentence in enumerate(sentences):
    sentences[i] = [vocab.loc[word] for word in sentence]

from nltk.util import skipgrams

window_size = 10

data = []
for sentance in sentences:
    data += skipgrams(sentance, 2, window_size)

data = pd.DataFrame(data, columns=['x', 'y'])
print(data.head())    

validation_size = 5000

data_valid = data.iloc[-validation_size:]
data_train = data.iloc[:-validation_size]
print('Train size:', len(data_train), 'Validation size:', len(data_valid))

learning_rate = .01
embed_size = 300
batch_size = 64
steps = 1000

inputs = tf.placeholder(tf.int32, [None])
targets = tf.placeholder(tf.int32, [None])

embeddings = tf.Variable(tf.random_uniform((len(vocab), embed_size), -1, 1))
embed = tf.nn.embedding_lookup(embeddings, inputs)


logits = tf.layers.dense(embed, len(vocab), activation=None,
    kernel_initializer=tf.random_normal_initializer())

labels = tf.one_hot(targets, len(vocab))
loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
loss = tf.reduce_mean(loss)

train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

def get_batches(x, y, batch_size, n=None):
    if n:
        # cheap way to add some randomization
        rand_start = np.random.randint(0, len(x) - batch_size * n)
        x = x[rand_start:]
        y = y[rand_start:]

    for start in range(len(x))[::batch_size][:n]:
        end = start + batch_size
        yield x[start:end], y[start:end]

step = 0
while step < steps:
    start = time.time()
    
    # shuffle train data once in while
    if step % 1000 == 0:
        data_train = data_train.sample(frac=1.)
    
    # train part
    train_loss = []
    for x, y in get_batches(
        data_train['x'].values, data_train['x'].values, batch_size, n=10000):
        step += 1
        _, batch_loss = sess.run([train_op, loss], {inputs: x, targets: y})
        train_loss.append(batch_loss)

    # validation prat (one batch of "validation_size")
    feed_dict = {inputs: data_valid['x'].values, targets: data_valid['x'].values}
    valid_loss, x_vectors = sess.run([loss, embed], feed_dict)
    y_vectors = sess.run(embed, {inputs: data_valid['x'].values})

    # outputs
    print('Step:', step, 'TLoss:', np.mean(train_loss), 'VLoss:', np.mean(valid_loss),
          'Similarity: %.3f' % cosine_similarity(x_vectors, y_vectors).mean(),
          'Seconds %.1f' % (time.time() - start))



vectors = sess.run(embeddings)
vectors = pd.DataFrame(vectors, index=vocab.index)

from sklearn.metrics.pairwise import cosine_similarity

print('Similarity:')
print('   computer to mouse =', cosine_similarity(vectors.loc[['computer']], vectors.loc[['mouse']])[0][0])
print('   cat to mouse =', cosine_similarity(vectors.loc[['cat']], vectors.loc[['mouse']])[0][0])
print('   dog to mouse =', cosine_similarity(vectors.loc[['dog']], vectors.loc[['mouse']])[0][0])

