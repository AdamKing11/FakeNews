from datetime import datetime as dtime
import pandas as pd
import numpy, pickle, sys, re

from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn import svm

# for related vs unrelated
from related import tfidf_transform_data, tfidf_predictions, list_of_terms, build_vocab
# for classifying agree vs disagree vs discuss
from agree import build_training, split_train_test

def pnow():
	print('\t\t', dtime.now())


if __name__ == '__main__':

	with_stemming = False
	with_lemmatization = False

	### loading
	pnow()
	print('loading....')
	# load in the bodies
	bodies = pd.read_csv("train_bodies.csv", doublequote=True)
	
	# load in the training and testing stances
	trains = pd.read_csv("train_stances_csc483583.csv")
	tests = pd.read_csv("test_stances_csc483583.csv")
	# combine them so we can build a BIG list with all terms (so we can
	# compare any headline/document pair)
	all_stances = trains.append(tests, ignore_index = True)


	# get a list of terms from all documents/headlines
	article_terms = list_of_terms(bodies['articleBody'].str.lower().tolist(), 
		do_stem = with_stemming, do_lemmatize = with_lemmatization)
	head_terms = list_of_terms(all_stances['Headline'].str.lower().tolist(),
		do_stem = with_stemming, do_lemmatize = with_lemmatization)

	# build a vocabulary using ALL terms from both
	vocab = build_vocab(article_terms, head_terms)
	pnow()
	print('done loading.')

	# now we split into training/testing pandas dataframes
	# we do it this way to get UNIQUE rowIDs for each headline-docID pair
	train_stances = all_stances[:len(trains)]
	test_stances = all_stances[len(trains):]
	
	# clear up some memory...
	del trains
	del tests
###################
#
###################

	### build classifiers
	# related vs unrelated
	pnow()
	print('building tf-idf for related vs unrelated')
	Body_tfidf, Head_tfidf = tfidf_transform_data(article_terms, head_terms, vocab)
	# disagree vs discuss/agree
	pnow()
	print('building svm for disagree vs discuss/agree')
	Xs, ys, disagree_rowIDs = build_training(bodies, train_stances, 
		article_terms, head_terms, 
		do_stem = with_stemming, do_lemmatize = with_lemmatization, 
		class_merge = {'agree' : 'discuss'})
	print('fitting svm - rbf')
	pnow()
	disagree_svm = svm.SVC(kernel='rbf')
	disagree_svm.fit(Xs, ys)
	# agree vs discuss
	print('building svm for discuss vs agree')
	Xs, ys, agr_rowIDs = build_training(bodies, train_stances, 
		article_terms, head_terms, 
		do_stem = with_stemming, do_lemmatize = with_lemmatization, 
		class_ignore = ['disagree', 'unrelated'])
	
	print('fitting svm - rbf')
	discuss_svm = svm.SVC(kernel='rbf')
	discuss_svm.fit(Xs, ys)
	pnow()
	print('\n\ndone training.', len(test_stances), 'headline-article pairs to classify.')

###################
#
###################

	# test_stances for the test data!!!
	predictions, gold, rel_rowIDs = tfidf_predictions(Body_tfidf, Head_tfidf, bodies, test_stances)
	#print('related vs unrelated:')
	#print(classification_report(gold, predictions))
	all_predictions = {}
	# get all the doc-headline pairs the classifier determined were "related" (discuss, agree, disagree)
	predicted_related = []
	for i, rID in enumerate(rel_rowIDs):
		if predictions[i] == 'related':
			predicted_related.append(rID)
		else:
			all_predictions[rID] = 'unrelated'

	print('done with related vs unrelated.')
	# get a subset of the stances for head/doc pairs tf-idf said were "related"
	related_stances = all_stances.ix[predicted_related]
	# take all the things labeled as "related" and put them through the next level of the classifier
	Xs, _, disagree_rowIDs = build_training(bodies, related_stances, 
		article_terms, head_terms, 
		do_stem = with_stemming, do_lemmatize = with_lemmatization, 
		class_merge = {'unrelated' : 'discuss'})
	# use the SVM we trained to make predictions
	predictions = disagree_svm.predict(Xs)

	predicted_discuss = []
	# go through the predictions. if we predict it as "disagree", save it
	# it we predict it as discuss, take it to the next step and classify between "discuss" and "agree"
	for i, rID in enumerate(disagree_rowIDs):
		if predictions[i] == 'discuss':
			predicted_discuss.append(rID)
		else:
			all_predictions[rID] = 'disagree'

	print('done with disagree vs discuss/agree.')
	# get a sub-dataframe of only the headline/articles we think are agree/discuss
	discuss_stances = related_stances.ix[predicted_discuss]
	# put them in the right vectorized format for classification
	Xs, _, discuss_rowIDs = build_training(bodies, discuss_stances, 
		article_terms, head_terms, 
		do_stem = with_stemming, do_lemmatize = with_lemmatization, 
		class_merge = {'unrelated' : 'discuss', 'disagree' : 'discuss'})
	# note here! if any doc's have snuck through and AREN'T "discuss" or "agree", just call them "dicuss"
	predictions = discuss_svm.predict(Xs)
	
	# store the predicted label
	for i, rID in enumerate(discuss_rowIDs):
		all_predictions[rID] = predictions[i]
	print('done with discuss vs agree.')
	
	predictions = []
	gold = []
	for i, row in test_stances.iterrows():
		predictions.append(all_predictions[i])
		gold.append(row['Stance'])

###################
#
###################
	# look at F1 score overall and save the data 
	# so we can use the **official** scoring function
	print(classification_report(gold, predictions))
	if with_stemming:
		with open('ak_fn_stemming.csv', 'w') as wf:
			for x in range(len(gold)):
				wf.write(predictions[x] + '\t' + gold[x] + '\n')
	if with_lemmatization:
		with open('ak_fn_lemmatization.csv', 'w') as wf:
			for x in range(len(gold)):
				wf.write(predictions[x] + '\t' + gold[x] + '\n')
	else:
		with open('ak_fn_tokens.csv', 'w') as wf:
			for x in range(len(gold)):
				wf.write(predictions[x] + '\t' + gold[x] + '\n')