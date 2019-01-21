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

cores = multiprocessing.cpu_count()

# reviews_df = pd.read_csv('data/amazon_reviews_us_Electronics_v1_00.tsv', sep='\t', encoding="utf8")
# print(reviews_df.head())
# toDatabase = [] 
# with open('data/amazon_reviews_us_Electronics_v1_00.tsv', "rt",encoding="utf8") as amazonReviews:
#     amazonReviews = csv.reader(amazonReviews, delimiter='\t')

#     #loop over all the records
#     count = 0
#                                    #Storing actual values in a dictionary
#     for amazonReview in amazonReviews:
#         count += 1
#         if count == 1:
#             keyArray = amazonReview
#             print(keyArray)
#         else:
#             reviewDictionary = {}
#             for i in range(0, len(keyArray)):
#                 key = keyArray[i]
#                 try:
#                     value = amazonReview[i]
#                     if key == 'review_date':
#                     	val = value
#                     	value = datetime.strptime(str(value), '%Y-%m-%d')
#                     reviewDictionary[key] = value
#                 except:
#                     pass
#             toDatabase.append(reviewDictionary)
# print(val)
# print(toDatabase[0])
# print(len(toDatabase))

# reviews_df = pd.DataFrame(toDatabase)
# print(reviews_df.head())

# reviews_headphones = reviews_df[reviews_df.product_title.str.contains(re.compile("headphone|earphone|headphones|earphones", re.IGNORECASE))]
# print(reviews_headphones.head())
# print(len(reviews_headphones))
# print('--------------------')
# reviews_headphones = reviews_headphones[~reviews_headphones.product_title.str.contains(re.compile("case|tips|hanger", re.IGNORECASE))]
# print(reviews_headphones.head())
# print(len(reviews_headphones))
# reviews_headphones.to_csv("data/headphones_review.csv", index=False)

# dataset = pd.read_csv("data/headphones_review.csv",encoding = "utf-8")
# print(dataset.head())
# print(len(dataset))
# dataset['review_body'] = dataset['review_body'].apply(lambda x:re.sub('[^a-zA-Z]', ' ',str(x)))
# print(dataset.head())
# dataset.to_csv("data/headphones_review_cleaned.csv", index=False,encoding = "utf-8")

# for i, line in enumerate(dataset['review_body']):
#     # For training data, add tags
#     all_x_w2v.append(TaggedDocument(gensim.utils.simple_preprocess(line), str(i)))

# all_x_w2v = []
# dataset = pd.read_csv("data/headphones_review_cleaned2.csv",encoding="utf-8")
# # dataset['review_body'] = dataset['review_body'].apply(lambda x:re.sub(' +', ' ', str(x)))


# for index, row in dataset.iterrows():
#     all_x_w2v.append(TaggedDocument(nltk.word_tokenize(str(row['text']).lower()), str(index)))


# print(all_x_w2v[0])
# print(len(all_x_w2v))

# with open('dataset_headphones.pkl','wb') as f:
#   pickle.dump(all_x_w2v,f)

with open('dataset_headphones.pkl', 'rb') as f:
    all_x_w2v = pickle.load(f)

# model_ug_dbow = Doc2Vec(dm=0, size=100, negative=5, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
# model_ug_dbow.build_vocab([x for x in tqdm(all_x_w2v)])

# for epoch in range(30):
#     model_ug_dbow.train(utils.shuffle([x for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
#     model_ug_dbow.alpha -= 0.002
#     model_ug_dbow.min_alpha = model_ug_dbow.alpha

# model_ug_dbow.save("dbow.model")
# print("Model Saved")

model= Doc2Vec.load("dbow.model")

doc_id = random.randint(0, len(all_x_w2v) - 1)
# doc_id = 0
inferred_vector = model.infer_vector(all_x_w2v[doc_id][0])
print(inferred_vector)
sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

print('Test Document ({}): «{}»\n'.format(doc_id, ' '.join(all_x_w2v[doc_id][0])))
print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n')
for label, index in [('MOST', 0), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
    print(sims[index])
    print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(all_x_w2v[int(sims[index][0])][0])))
