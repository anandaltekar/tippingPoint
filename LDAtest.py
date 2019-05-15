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
from pprint import pprint 
from random import sample

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec, Phrases
from gensim.models.phrases import Phraser
from gensim.models.doc2vec import LabeledSentence,TaggedDocument
import multiprocessing
from sklearn import utils
from sklearn.linear_model import LogisticRegression
from collections import Counter
from nltk.corpus import stopwords
import spacy
from gensim import corpora
from sentimentLDA import *
import os
import urllib
import tarfile
vocabSize = 50000


# with open('docs_stem.pkl', 'rb') as f:
# 	documentList = pickle.load(f)
# documentList = documentList[:50000]
# dictionary = corpora.Dictionary(documentList)
# doc_term_matrix = [dictionary.doc2bow(doc) for doc in tqdm(documentList)]
# LDA = gensim.models.ldamodel.LdaModel
# print(len(doc_term_matrix))
# lda_model = LDA(corpus=doc_term_matrix,
#                 id2word=dictionary,
#                 num_topics=10, 
#                 random_state=100,
#                 chunksize=1000,
#                 passes=50)
# print("####################Topics##################################")
# pprint(lda_model.print_topics())
def sent_to_words(sentences):
	for sentence in sentences:
		yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

sentences = []
document_sentences = []
reviews = []
dataset = pd.read_csv("data/headphones_review2.csv",encoding="latin",engine='python')
dataset = dataset.sample(n=10000, random_state=1)
for i, line in tqdm(enumerate(dataset['review_body'])):
	if len(str(line)) > 3:
		reviews.append(str(line))
		doc_sent = nltk.sent_tokenize(str(line))
		sentence_words = list(sent_to_words(doc_sent))
		for word in sentence_words:
			if len(word) > 3:
				sentences.append(word)
		document_sentences.append(sentence_words)
print(len(sentences))
print(sentences[0])
print(len(reviews))
print(reviews[0])

sentences = sample(sentences,10000)

# print(document_sentences[0])
# print(len(document_sentences))
# print(len(document_sentences[0]))

sampler = SentimentLDAGibbsSampler(10, 0.1, 0.01, 0.3)
sampler.run(sentences, 100, "Sent_LDA.pkl", False)

sampler.getTopKWords(25)
print("#############################################")
sampler.getTopKWordsByLikelihood(25)
















	