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
# nltk.download('punkt')
# from gensim.models import Doc2Vec, Phrases
# from gensim.models.phrases import Phraser
# from gensim.models.doc2vec import LabeledSentence,TaggedDocument
# import multiprocessing
# from sklearn import utils
# from sklearn.linear_model import LogisticRegression
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
import spacy
from gensim import corpora

dataset = pd.read_csv("data/headphones_review2.csv",encoding="latin",engine='python')

def cleaning(dataset):
	

# cleaning(dataset)