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


with open('docs_stem.pkl', 'rb') as f:
	documentList = pickle.load(f)
documentList = documentList[:50000]
dictionary = corpora.Dictionary(documentList)
doc_term_matrix = [dictionary.doc2bow(doc) for doc in tqdm(documentList)]
LDA = gensim.models.ldamodel.LdaModel
print(len(doc_term_matrix))
lda_model = LDA(corpus=doc_term_matrix,
                id2word=dictionary,
                num_topics=10, 
                random_state=100,
                chunksize=1000,
                passes=50)
print("####################Topics##################################")
pprint(lda_model.print_topics())















	