#!/usr/bin/env python
# -*- coding: utf-8 -*-
#python 3.6.0
import os
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import stop_words
import string
import nltk
np.seterr(divide='ignore', invalid='ignore')

#read the input file for word2vec training
documents = list()
with open('/Users/agazi/Thesis/DATA_MasterThesis/Query.txt') as file:
	for line in file:
		line.strip()
		documents.append(line.lower())
#print (documents)
#create dictionary for lemmatization
lemma={}
for line in open('/Users/agazi/Thesis/DATA_MasterThesis/lemmatization-sv.txt', 'r'):
    split = line.strip().split('\t', 1)
    lemma[split[1]] = split[0]
#print (lemma['A:et'])

# #remove stopwords, punctuaiton, numbers and lemmatize
# #stoplist = set(line.strip() for line in open('/Users/agazi/Thesis/stopwords/stopword_sw.txt'))
stoplist = set()
stoplist.update(['.', ',', '"', '?', '!', ';', '(', ')', '[', ']', '{', '}', '+','#', '/'])
#include the number if necessary
#'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

statements = [[lemma.get(word,word) for word in document.split() if lemma.get(word,word) not in stoplist]
 			for document in documents]

#train the model, save the model
model = Word2Vec(statements, size=100, window=5, min_count=1,workers=4)
model.save("model_1.mod") 

#load previously trained model
#model = Word2Vec.load("model_1.mod")

set1 = set(line.strip() for line in open('/Users/agazi/Thesis/DATA_MasterThesis/Query.txt'))

vocabulary1 = [lemma.get(w,w) for w in set1 if lemma.get(w,w) in model.vocab and lemma.get(w,w) not in stoplist]
vocabulary = list(set(vocabulary1))


# d = []
# with open('/Users/agazi/Thesis/DATA_MasterThesis/Query.txt') as file:
# 	for line in file:
# 		line = line.strip()
# 		line = line.translate(str.maketrans(' ',' ',string.punctuation))  # Removes the ? and other punctiuations from words for python 3
# 		d.append(line)

# vect = CountVectorizer(vocabulary=vocabulary).fit(d)

# from sklearn.metrics import euclidean_distances
# W_ = np.array([model[w] for w in vect.get_feature_names() if w in model])
# D_ = euclidean_distances(W_)
# D_ = D_.astype(np.double)
# D_ /= D_.max()  # just for comparison purposes

# from scipy.spatial.distance import cosine
# v_array = np.array(vect.transform(d))
# print(v_array.shape)
# print (type (v_array))
# print (v_array.size)
# # v_1 = v_1.toarray().ravel()
# # print("cosine(doc_1, doc_2) = {:.9f}".format(cosine(v_1, v_2)))

# a = np.array([['agazi','1'], ['Berihu','2'], ['Mekonnen', '3']])
# print (a.size)
# from pyemd import emd
# # pyemd needs double precision input
# # v_1 = v_1.astype(np.double)
# # v_1 /= v_1.sum()
# # print("d(doc_1, doc_2) = {:.9f}".format(emd(v_1, v_2, D_)))