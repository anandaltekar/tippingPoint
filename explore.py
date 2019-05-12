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

SentencesList = []

from nltk.tokenize import TweetTokenizer, sent_tokenize

def cleaning(dataset):

	tokenizer_words = TweetTokenizer()
	stop_words=["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than","too", "very", "s", "t", "can", "will", "just", "don", "should", "now"] + list(stopwords.words('english'))

	stemmer = PorterStemmer()

	for i, line in tqdm(enumerate(dataset['review_body'])):
		tokens_sentences = [tokenizer_words.tokenize(str(t)) for t in nltk.sent_tokenize(str(line))]
		tokens_sentences =  [[re.sub('[^a-zA-Z]+', '',str(word)) for word in sentence] for sentence in tokens_sentences]
		tokens_sentences = [[word.lower() for word in sentence] for sentence in tokens_sentences]
		tokens_sentences =  [[stemmer.stem(word) for word in sentence if (len(word) > 2 and word[:3] != "htt" and word not in stop_words)] for sentence in tokens_sentences]
		tokens_sentences = [t for t in tokens_sentences if len(t) > 2]	
		SentencesList.append(tokens_sentences)

	print(SentencesList[0:5])
	with open('Sentences_stem.pkl', 'wb') as f:
	    pickle.dump(SentencesList,f)

# cleaning(dataset)

# with open('Sentences_stem.pkl', 'rb') as f:
#     SentencesList =  pickle.load(f)
with open('docs_stem.pkl', 'rb') as f:
	documentList = pickle.load(f)
# with open('words_stem.pkl', 'rb') as f:
# 	wordList = pickle.load(f)





to_remove = [i for i, val in enumerate(documentList) if len(val)<=2]
for index in reversed(to_remove): # start at the end to avoid recomputing offsets
    del documentList[index]
    # del SentencesList[index]




print(len(documentList))
# print(len(SentencesList))
# print(len(wordList))



dictionary = corpora.Dictionary(documentList)
dictionary.filter_extremes(no_below=5)

with open('docs_stem.pkl', 'wb') as f:
	pickle.dump(documentList,f)

with open('dictionary_stem.pkl', 'wb') as f:
	pickle.dump(dictionary,f)


sentiWordsList = []
with open("pos.txt") as file:
    words = [line.strip() for line in file]
    sentiWordsList.append(words)

with open("neg.txt") as file:
    words = [line.strip() for line in file]
    sentiWordsList.append(words)

# with open('docs_stem.pkl', 'wb') as f:
# 	pickle.dump(documentList,f)
# with open('words_stem.pkl', 'wb') as f:
# 	pickle.dump(wordList,f)
# with open('dict_stem.pkl', 'wb') as f:
# 	pickle.dump(dictionary,f)
# # with open('dict_stem.pkl', 'wb') as f:
# # 	pickle.dump(dictionary,f)
# with open('doc_term_stem.pkl', 'wb') as f:
# 	pickle.dump(doc_term_matrix,f)
#
def calculatePhi(sent_word_topic,sum_sent_word_topic,betas,sumBeta,sentiWordsList):
	print("Calculating PHI....")
	numSenti = len(sent_word_topic)
	numWords = len(sent_word_topic[0])
	numTopics = len(sent_word_topic[0][0])
	
	Phi = np.zeros((numSenti,numWords,numTopics))
	wordLexicons = [-1 for i in range(0,numWords)]
	for w in range(0,numWords):
		for s in range(0,numSenti):
			if (w in sentiWordsList[s]):
				wordLexicons[w] = s
	for s in range(0,numSenti):
		for w in range(0,numWords):
			for t in range(0,numTopics):
				beta = 0
				if (wordLexicons[w] == -1):
					beta = betas[0]
				elif (wordLexicons[w] == s): 
					beta = betas[1]
				else:
					beta = betas[2]
				
				value = (sent_word_topic[s][w][t] + beta) / (sum_sent_word_topic[s][t] + sumBeta[s])
				Phi[s][w][t] = value
	return Phi

def calculateTheta(sent_doc_topic, sum_sent_doc_topic, alpha, sumAlpha):
	print("Calculating Theta...")
	numSenti = len(sent_doc_topic)
	numDocs = len(sent_doc_topic[0])
	numTopics = len(sent_doc_topic[0][0])
	
	Theta = np.zeros((numSenti,numDocs,numTopics))

	for s in range(0,numSenti):
		for d in range(0,numDocs):
			for t in range(0,numTopics): 
				if (sum_sent_doc_topic[d][s] + sumAlpha) > 0:
					value = (sent_doc_topic[s][d][t] + alpha) / (sum_sent_doc_topic[d][s] + sumAlpha)
				else:
					value = 0
				Theta[s][d][t] = value
	return Theta

def calculatePi(matrixDS, sumDS, gammas, sumGamma):
	print("Calculating Pi...")
	numDocs = len(matrixDS)
	numSenti = len(matrixDS[0])
	
	Pi = np.zeros((numDocs, numSenti))
	
	for d in range(0,numDocs):
		for s in range(0,numSenti):
			value = (matrixDS[d][s] + gammas[s]) / (sumDS[d] + sumGamma);
			Pi[d][s] = value
	
	return Pi

def getSortedColIndex(col, n,this_list):
	sortedList = []
	
	for i in range(0,n):
		maxValue = -999999;
		maxIndex = -1
		for row in range(0,len(this_list)):
			if(this_list[row][col] > maxValue):
				exist = False
				for j in range(0,len(sortedList)):
					if (sortedList[j] == row):
						exist = True
						break
				if(exist == False):
					maxValue = this_list[row][col]
					maxIndex = row;
		sortedList.append(maxIndex);
	
	return sortedList

def ASUM(dictionary,sentiWordsList):

	# dictionary = corpora.Dictionary(documentList)
	# dictionary.filter_extremes(no_below=5)
	# SentenceDictList = [[{'text':dictionary.doc2idx(sent),'num_sent':-1,'topic':-1,'sent':-1} for sent in doc] for doc in SentencesList]
	# SentenceDictList = [['text' : [word for word in sent['text'] if word != -1] for sent in doc] for doc in SentenceDictList]
	with open('sentence_dict_stem.pkl', 'rb') as f:
		SentenceDictList = pickle.load(f)
	# SentenceDictList = SentenceDictList[:100]
	WordDictList = []
	for i in range(0,len(dictionary)):
		WordDictList.append({'sent':-1,'topic':-1})
	# for doc in SentenceDictList:
	# 	count = {}
	# 	words = []
	# 	for sentence in doc:
	# 		for word in sentence['text']:
	# 			if word != -1:
	# 				if str(word) not in count:
	# 					count[str(word)] = 1
	# 				else:
	# 					count[str(word)] += 1
	# 				words.append(word)
	# 		sentence['count'] = count
	# 		sentence['text'] = words
	num_sent_words = 0
	for words in sentiWordsList:
		num_sent_words += len(words)
	print(num_sent_words)
	print(len(WordDictList))
	print(len(SentenceDictList))
	print(SentenceDictList[-1][0]['text'])

	# with open('sentence_dict_stem.pkl', 'wb') as f:
	# 	pickle.dump(SentenceDictList,f)
	# with open('dictionary_stem.pkl', 'wb') as f:
	# 	pickle.dump(dictionary,f)

	n_sent = 2
	n_topics = 30
	num_documents = len(SentenceDictList)
	num_words = len(dictionary)
	sent_word_topic = np.zeros([n_sent,num_words,n_topics])
	sum_sent_word_topic = np.zeros([n_sent,n_topics])
	sent_doc_topic = np.zeros([n_sent,num_documents,n_topics])
	sum_sent_doc_topic = np.zeros([num_documents,n_sent])
	doc_sent = np.zeros([num_documents,n_sent])
	sum_doc = np.zeros([num_documents])



	for i,doc in enumerate(SentenceDictList):
		for sentence in doc:
			new_sent = -1
			numSentenceSenti = 0;
			for word in sentence['text']:
				for s,sentList in enumerate(sentiWordsList):
					if word != -1 and dictionary.get(word) in sentList:
						if (numSentenceSenti == 0 or s != new_sent):
							numSentenceSenti += 1;
						WordDictList[word]['sent'] = s;
						new_sent = s;
			sentence['num_sent'] = numSentenceSenti;

			if sentence['num_sent'] == -1:
				new_sent =  random.randint(0, n_sent-1) 
			new_topic = random.randint(0, n_topics-1) 

			if(numSentenceSenti <= 1):
				sentence['topic'] = new_topic
				sentence['sent'] = new_sent

				for word in sentence['text']:
					WordDictList[word]['sent'] = new_sent
					WordDictList[word]['topic'] = new_topic
					sent_word_topic[new_sent][word][new_topic] += 1
					sum_sent_word_topic[new_sent][new_topic] += 1

				sent_doc_topic[new_sent][i][new_topic] += 1
				doc_sent[i][new_sent] += 1

				sum_sent_doc_topic[i][new_sent] += 1
				sum_doc[i]+=1

	gammas = np.ones((n_sent),dtype=np.double)
	betas = [0.001,0.001,0]
	alpha = 0.1
	num_iter = 100

	sum_alpha = alpha*n_topics
	sumBetaCommon = betas[0] * (num_words - num_sent_words)
	sumBeta = np.zeros((n_sent))

	for s in range(0,n_sent):
		sent_words_count = len(sentiWordsList[s])
		sumBeta[s] = sumBetaCommon + betas[1]*sent_words_count + betas[2]*(num_sent_words - sent_words_count)

	sum_gamma = np.sum(gammas)
	probTable = np.zeros((n_topics,n_sent))
	sum_prob = 0

	print("ITER--->")
	for idx in tqdm(range(0,num_iter)):
		for i,doc in enumerate(SentenceDictList):
			for sentence in doc:
				if sentence['sent'] == -1:
					continue
				old_topic = sentence['topic']
				old_sent = sentence['sent']

				sent_doc_topic[old_sent][i][old_topic] -= 1
				doc_sent[i][old_sent] -= 1

				sum_sent_doc_topic[i][old_sent] -= 1
				sum_doc[i] -= -1

				for word in sentence['text']:
					sent_word_topic[old_sent][word][old_topic] -= 1
					sum_sent_word_topic[old_sent][old_topic] -= 1

			for s in range(0,n_sent):
				trim = False

				for word in sentence['count'].keys():
					if WordDictList[int(word)]['sent'] != s:
						trim = True
						break
				
				if(trim):
					for t in range(0,n_topics):
						probTable[t][s] = 0
				
				else:
					for t in range(0,n_topics):
						beta0 = sum_sent_word_topic[s][t] + sumBeta[s]
						m0 = 0
						expected_tsw = 1
						for word in sentence['count'].keys():
							beta = 0
							if(WordDictList[int(word)]['sent'] == -1):
								beta = betas[0]
							elif(WordDictList[int(word)]['sent'] == s):
								beta = betas[1]
							else:
								beta = betas[2]
							betaw = sent_word_topic[s][int(word)][t] + beta
							this_word_count = sentence['count'][word]

							for m in range(0,this_word_count):
								expected_tsw *= (betaw + m) / (beta0 + m0)
								m0+=1 
					if (sum_sent_doc_topic[i][s] + sum_alpha) > 0:
						probTable[t][s] = (sent_doc_topic[s][i][t] + alpha) / (sum_sent_doc_topic[i][s] + sum_alpha) * (doc_sent[i][s] + gammas[s]) * expected_tsw
					else:
						probTable[t][s] = 0
					sum_prob += probTable[t][s]
			new_topic = 0
			new_sent = 0
			randNo = random.random() * sum_prob
			found = False
			tmpSumProb = 0
			for t in range(0,n_topics):
				for s in range(0, n_sent):
					tmpSumProb += probTable[t][s];
					if(randNo <= tmpSumProb):
						new_topic = t
						new_sent = s
						found = True
					if(found):
						break
				if(found):
					break
			sentence['sent'] = new_sent
			sentence['topic'] = new_topic
			for word in sentence['text']:
				WordDictList[word]['sent'] = new_sent
				WordDictList[word]['topic'] = new_topic
				sent_word_topic[new_sent][word][new_topic] += 1
				sum_sent_word_topic[new_sent][new_topic] += 1

			sent_doc_topic[new_sent][i][new_topic] += 1
			doc_sent[i][new_sent] += 1
			
			sum_sent_doc_topic[i][new_sent] += 1
			sum_doc[i] += 1

		Phi = calculatePhi(sent_word_topic,sum_sent_word_topic,betas,sumBeta,sentiWordsList)
		Theta = calculateTheta(sent_doc_topic, sum_sent_doc_topic, alpha, sum_alpha)
		Pi = calculatePi(doc_sent, sum_doc, gammas, sum_gamma)

	Phi = calculatePhi(sent_word_topic,sum_sent_word_topic,betas,sumBeta,sentiWordsList)
	Theta = calculateTheta(sent_doc_topic, sum_sent_doc_topic, alpha, sum_alpha)
	Pi = calculatePi(doc_sent, sum_doc, gammas, sum_gamma)

	with open('phi_1.pkl', 'wb') as f:
		pickle.dump(Phi,f)

	with open('theta_1.pkl', 'wb') as f:
		pickle.dump(Theta,f)

	with open('pi_1.pkl', 'wb') as f:
		pickle.dump(Pi,f)

	for s in range(0,n_sent):
		for t in range(0,n_topics):
			print("S"+s+"-T"+t+",")
	print("\n")
	num_Prob_Words = 100
	wordIndices = np.zeros((n_sent,n_topics,num_Prob_Words))
	for s in range(0,n_sent):
		for t in range(0,n_topics):
			sortedIndexList = getSortedColIndex(t, num_Prob_Words,Phi[s])
			for t in range(0,len(sortedIndexList)):
				wordIndices[s][t][w] = sortedIndexList[w]
	for w in range(0,num_Prob_Words): 
		for s in range(0,n_sent): 
			for t in range(0,n_topics):
				index = wordIndices[s][t][w]
				print(dictionary.get(index)+" ("+Phi[s][index][t]+"),")
		print("\n")

ASUM(dictionary,sentiWordsList)