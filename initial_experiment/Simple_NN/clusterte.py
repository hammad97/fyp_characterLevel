import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 1000000

def create_lexicon(C1,C2,C3,C4,C5):
    lexicon = []
    for fi in [C1,C2,C3,C4,C5]:
        	with open(fi,'r') as f:
        		contents = f.readlines()
        		for l in contents[:hm_lines]:
        			all_words = word_tokenize(l)
        			lexicon += list(all_words)
    
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
    w_counts = Counter(lexicon)
    l2 = []
    for w in w_counts:
        if 100 > w_counts[w] > 10:
            l2.append(w)
    print(len(l2))
    return l2
                    
def sample_handling(sample,lexicon,classification):
    featureset = []
    
    with open(sample,'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower())
                    features[index_value] += 1
                    
            features = list(features)
            featureset.append([features,classification])
            
    return featureset

def create_feature_sets_and_labels(C1,C2,C3,C4,C5,test_size = 0.1):
    lexicon = create_lexicon(C1,C2,C3,C4,C5)
    features = []
    features += sample_handling('C1.txt',lexicon,[1,0,0,0,0])
    features += sample_handling('C2.txt',lexicon,[0,1,0,0,0])
    features += sample_handling('C3.txt',lexicon,[0,0,1,0,0])
    features += sample_handling('C4.txt',lexicon,[0,0,0,1,0])
    features += sample_handling('C5.txt',lexicon,[0,0,0,0,1])
    random.shuffle(features)
    features = np.array(features)
    
    testing_size = int(test_size*len(features))
    
    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])
    
    return train_x,train_y,test_x,test_y

if __name__ == '__main__':
	train_x,train_y,test_x,test_y = create_feature_sets_and_labels('C1.txt','C2.txt','C3.txt','C4.txt','C5.txt')
	# if you want to pickle this data:
	with open('sentiment_set.pickle','wb') as f:
		pickle.dump([train_x,train_y,test_x,test_y],f)
