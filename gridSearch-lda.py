import numpy as np
import pandas as pd
import re, nltk, spacy, gensim

from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

#plotting
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt
#%matplotlib inline

df = pd.read_csv("/Users/anand/Downloads/headphones_review2.csv", encoding="latin")
df = df.sample(frac=0.05, random_state=99)

#CONVERT TO LIST
data = df['review_body']
#print(data[:5])

#TOKENIZE AND CLEAN-UP
def sent_to_words(sentences):
    for sentence in sentences:
    	#print(sentence)
    	yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))
        # deacc=True removes punctuations

#data_words = []
data_words = list(sent_to_words(data))
print(data_words[:2])
print(len(data_words))

#LEMMATIZATION
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
	"""https://spacy.io/api/annotation"""
	texts_out = []
	for sent in texts:
		doc = nlp(" ".join(sent))
		texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
	return texts_out

nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only Noun, Adj, Verb, Adverb
data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[:2])
print(len(data_lemmatized))

#CREATE THE DOCUMENT WORD MATRIX

vectorizer = CountVectorizer(
							analyzer = 'word', 
							min_df = 1,						  # Min reqd occurences
							stop_words = 'english',			  # Remove stop words
							lowercase = True,				  # Convert to lowercase
							token_pattern = '[a-zA-Z]{3,}' # num of chars > 3, max features = 50000, max no. of unique words
							)
data_vectorized = vectorizer.fit_transform(data_lemmatized)

#CHECK THE SPARSITY

# Materialize the sparse data
data_dense = data_vectorized.todense()

# Compute Sparsicity = Percentage of Non-Zero cells
print("Sparsicity: ", ((data_dense > 0).sum()/data_dense.size)*100, "%")

#BUILD LDA MODEL WITH SKLEARN

lda_model = LatentDirichletAllocation(
										n_components = 10,
										max_iter = 100,
										learning_method = 'online',
										random_state = 100,
										batch_size = 250,
										evaluate_every = 5,	   #Compute perplexity every n itrns
										n_jobs = -1,			   #use all available CPUs 
									 )
lda_output = lda_model.fit_transform(data_vectorized)
print(lda_output)

#DIAGNOSE MODEL PERFORMANCE WITH PERPLEXITY AND LOG-LIKELIHOOD

#Log-likelihood higher the better
print("Log Likelihood: ", lda_model.score(data_vectorized))

# Perplexity: Lower the better. Perplexity = exp(-1. * log-likelihood per word)
print("Perplexity: ", lda_model.perplexity(data_vectorized))

# See model parameters
pprint(lda_model.get_params())

# Define Search Param
search_params = {'n_components': [5, 7, 10, 12], 'learning_decay': [.5, .7, .9]}

# Init the Model
lda = LatentDirichletAllocation()

# Init Grid Search Class
model = GridSearchCV(lda, param_grid=search_params)

# Do the Grid Search
model.fit(data_vectorized)

# Best Model
best_lda_model = model.best_estimator_

# Model Parameters
print("Best Model's Params: ", model.best_params_)

# Log Likelihood Score
print("Best Log Likelihood Score: ", model.best_score_)

# Perplexity
print("Model Perplexity: ", best_lda_model.perplexity(data_vectorized))

# Create Document - Topic Matrix
lda_output = best_lda_model.transform(data_vectorized)

# column names
topicnames = ["Topic" + str(i) for i in range(best_lda_model.n_topics)]

# index names
docnames = ["Doc" + str(i) for i in range(len(data))]

# Make the pandas dataframe
df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)

# Get dominant topic for each document
dominant_topic = np.argmax(df_document_topic.values, axis=1)
df_document_topic['dominant_topic'] = dominant_topic

# Styling
def color_green(val):
    color = 'green' if val > .1 else 'black'
    return 'color: {col}'.format(col=color)

def make_bold(val):
    weight = 700 if val > .1 else 400
    return 'font-weight: {weight}'.format(weight=weight)

# Apply Style
df_document_topics = df_document_topic.head(15).style.applymap(color_green).applymap(make_bold)
df_document_topics




