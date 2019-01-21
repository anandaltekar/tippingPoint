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

from collections import Counter
from nltk.corpus import stopwords

from gensim import corpora, models
import gensim

with open('data/output.txt', 'r',encoding="utf-8") as f:
    text = f.read()

articles = text.split('-----')
# articles = text
def stem_tokens(tokens):
    stemmed = []
    for item in tokens:
    	if item not in set(stopwords.words('english')):
        	# stemmed.append(stemmer.stem(item))
        	stemmed.append(item)
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens)
    return stems

all_words = []
for article in articles:
    article = re.sub('[^a-zA-Z]', ' ',article)
    all_words.append(tokenize(article.lower()))

# print(all_words)

# counts = Counter(all_words)
# most_occur = counts.most_common(100) 
  
# print(most_occur) 

dictionary = corpora.Dictionary(all_words)
    
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in all_words]

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=5, id2word = dictionary, passes=20)
print(ldamodel.print_topics(num_topics=5, num_words=5))
# print(counts[:10])