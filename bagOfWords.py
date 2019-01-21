import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import os
import nltk
from glob import glob
import pickle
import math
from datetime import datetime
import gensim
import csv
import random

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence,TaggedDocument
import multiprocessing
from sklearn import utils
from sklearn.linear_model import LogisticRegression
from collections import Counter
from nltk.corpus import stopwords
import spacy


all_x_w2v = []
dataset = pd.read_csv("data/headphones_review_cleaned2.csv",encoding="utf-8")

# def stem_tokens(tokens):
#     stemmed = []
#     for item in tokens:
#     	if item not in set(stopwords.words('english')):
#         	# stemmed.append(stemmer.stem(item))
#         	stemmed.append(item)
#     return stemmed

# def tokenize(text):
#     tokens = nltk.word_tokenize(text)
#     stems = stem_tokens(tokens)
#     return stems

# all_words = []
# for index, row in dataset.iterrows():
# 	all_words.append(tokenize(row['text'].lower()))

# with open('all_words.pkl','wb') as f:
#   pickle.dump(all_words,f)

# counts = Counter(all_words)
# most_occur = counts.most_common(10) 
  
# print(most_occur) 
# print(counts[:10])

dataset['text'] = dataset['text'].str.replace("[^a-zA-Z#]", " ")
# dataset['text'] = dataset.apply()
def remove_stopwords(rev):
    rev_new = " ".join([i for i in rev if i not in set(stopwords.words('english'))])
    return rev_new

# remove short words (length < 3)
dataset['text'] = dataset['text'].apply(lambda x: ' '.join([w for w in str(x).split() if len(w)>2]))

# remove stopwords from the text
reviews = [remove_stopwords(r.split()) for r in dataset['text']]

# make entire text lowercase
reviews = [r.lower() for r in reviews]

nlp = spacy.load('en', disable=['parser', 'ner'])

def lemmatization(texts, tags=['NOUN', 'ADJ']):
	output = []
	output_noun = []
	for sent in tqdm(texts):
		doc = nlp(" ".join(sent))
		tokenList = []
		tokenNounList = []
		for token in doc:
			token_lemma = token.lemma_
			if token.pos_ in tags:
				tokenList.append(token_lemma)
			if token.pos_ == 'NOUN':
				print(token_lemma)
				tokenNounList.append(token_lemma)
		output.append(tokenList)
		output_noun.append(tokenNounList)
	return output,output_noun

tokenized_reviews = pd.Series(reviews).apply(lambda x: x.split())
print(tokenized_reviews[1])
len(tokenized_reviews[1])

reviews_2,reviews_noun = lemmatization(tokenized_reviews)
print(reviews_2[1])
len(reviews_2[1])

print(reviews_noun[1])
len(reviews_noun[1])

with open('reviews2.pkl', 'wb') as f:
    pickle.dump(reviews_2,f)
with open('reviews_noun.pkl', 'wb') as f:
    pickle.dump(reviews_noun,f)

dictionary = corpora.Dictionary(reviews_2)
doc_term_matrix = [dictionary.doc2bow(rev) for rev in reviews_2]
LDA = gensim.models.ldamodel.LdaModel

lda_model = LDA(corpus=doc_term_matrix,
                id2word=dictionary,
                num_topics=7, 
                random_state=100,
                chunksize=1000,
                passes=50)

lda_model.print_topics()













	