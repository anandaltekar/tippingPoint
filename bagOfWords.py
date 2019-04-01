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


all_x_w2v = []
dataset = pd.read_csv("data/headphones_review_cleaned2.csv",encoding="utf-8")
dataset = dataset.sample(frac=0.25, random_state=99)

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
	stop_words=["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than","too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
	rev_new = " ".join([i for i in rev if i not in set(stopwords.words('english')) and i not in stop_words])
	return rev_new

# remove short words (length < 3)
dataset['text'] = dataset['text'].apply(lambda x: ' '.join([w for w in str(x).split() if len(w)>2]))

# remove stopwords from the text
reviews = [remove_stopwords(r.split()) for r in dataset['text']]

# make entire text lowercase
reviews = [r.lower() for r in reviews]

bigram = gensim.models.Phrases(pd.Series(reviews).apply(lambda x: x.split()), min_count=5, threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)

def make_bigrams(reviews):
    return [bigram_mod[doc] for doc in tqdm(reviews)]


with open('reviews_cleaned_sample.pkl', 'wb') as f:
    pickle.dump(reviews,f)

nlp = spacy.load('en')


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
				# print(token_lemma)
				tokenNounList.append(token_lemma)
		output.append(tokenList)
		output_noun.append(tokenNounList)
	return output,output_noun

tokenized_reviews = pd.Series(reviews).apply(lambda x: x.split())
tokenized_reviews = make_bigrams(tokenized_reviews)

with open('tokenized_reviews_bigrams_sample.pkl', 'wb') as f:
    pickle.dump(tokenized_reviews,f)

print(tokenized_reviews[1])
print(len(tokenized_reviews[1]))

reviews_2,reviews_noun = lemmatization(tokenized_reviews)
print(reviews_2[1])
print(len(reviews_2[1]))

print(reviews_noun[1])
print(len(reviews_noun[1]))

with open('reviews2_sample.pkl', 'wb') as f:
    pickle.dump(reviews_2,f)
with open('reviews_noun_sample.pkl', 'wb') as f:
    pickle.dump(reviews_noun,f)

# with open('reviews2_sample.pkl', 'rb') as f:
#     reviews_2 = pickle.load(f)
# with open('reviews_noun_sample.pkl', 'rb') as f:
#     reviews_noun = pickle.load(f)

dictionary = corpora.Dictionary(reviews_2)
doc_term_matrix = [dictionary.doc2bow(rev) for rev in tqdm(reviews_2)]
LDA = gensim.models.ldamodel.LdaModel
print(len(doc_term_matrix))
lda_model = LDA(corpus=doc_term_matrix,
                id2word=dictionary,
                num_topics=7, 
                random_state=100,
                chunksize=1000,
                passes=50)
print("####################Topics##################################")
pprint(lda_model.print_topics())

dictionary_noun = corpora.Dictionary(reviews_noun)
doc_term_matrix_noun = [dictionary_noun.doc2bow(rev) for rev in tqdm(reviews_noun)]
print(len(doc_term_matrix_noun))
# LDA = gensim.models.ldamodel.LdaModel

lda_model_noun = LDA(corpus=doc_term_matrix_noun,
                id2word=dictionary_noun,
                num_topics=7, 
                random_state=100,
                chunksize=1000,
                passes=50)
print("####################Topics NOUN##################################")
pprint(lda_model_noun.print_topics())
# lda_model_noun.print_topics()













	