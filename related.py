from datetime import datetime as dtime
import pandas as pd
import numpy, pickle, sys

from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report

from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer


def pnow():
	# print out current time
	print(dtime.now())

def list_of_terms(list_of_lists, tr = RegexpTokenizer(r'\w+'), do_stem = False, do_lemmatize = True):
	if do_stem:
		stemmer = SnowballStemmer("english", ignore_stopwords=True)
	elif do_lemmatize:
		lr = WordNetLemmatizer()

	term_list = []
	for l in list_of_lists:
		tokens = tr.tokenize(l)
		if do_stem:
			terms = [stemmer.stem(w) for w in tokens]
		elif do_lemmatize:
			terms = [lr.lemmatize(w) for w in tokens]
		else:
			terms = tokens
		term_list.append(terms)
	return term_list
####

def build_vocab(articles, heads):
	vocab = {}
		# get the terms and counts for the bodies
	for doc in articles:
		for l in doc:
			if l in vocab:
				vocab[l] += 1
			else:
				vocab[l] = 1
		# do the same for the headlines
	for h in heads:
		for l in h:
			if l in vocab:
				vocab[l] += 1
			else:
				vocab[l] = 1
	return vocab
####

def tfidf_transform_data(article_lems, head_lems, vocab):
	# build the tf-idf vectorizer
	transformer = TfidfVectorizer(vocabulary = vocab.keys(), min_df = 1, stop_words = 'english')
	Body_tfidf = transformer.fit_transform([' '.join(a) for a in article_lems])
	Head_tfidf = transformer.fit_transform([' '.join(h) for h in head_lems])

	return Body_tfidf, Head_tfidf


def tfidf_predictions(Body_tfidf, Head_tfidf, bodies_pd, stances_pd, threshold = .1):

	predictions = []
	gold = []
	rowIDs = []
	for i, row in stances_pd.iterrows():
		# get the index of the matching body article
		body_index = bodies_pd.loc[bodies_pd['Body ID'] == row['Body ID']].index[0]
	
		score = cosine_similarity(Body_tfidf[body_index], Head_tfidf[i])	
		if score[0] < threshold:
			predictions.append('unrelated')
		else:
			predictions.append('related')
	
		if row['Stance'] == 'unrelated':
			gold.append('unrelated')
		else:
			gold.append('related')

		rowIDs.append(i)
		if len(rowIDs) > 1000:
			#break
			pass
		print(len(rowIDs), end = '\r')
	
	print()
	return predictions, gold, rowIDs
####

#####################################################################
