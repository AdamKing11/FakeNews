from datetime import datetime as dtime
import pandas as pd
import numpy, pickle, sys, re

from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report

from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk import pos_tag, word_tokenize, ngrams

###############
refuting_words = ('fake', 'fraud', 'hoax', 'false', 'deny', 'doubt',
        'refute', 'despite', 'nope', 'doubt', 'bogus', 'debunk', 'prank',
        'retract', 'revoke', 'sometimes', 'rarely', 'misinformation', 'unusual',
        'viral', 'however', 'yet', 'though', 'wrong', 'blame', 'fake')

agree_words = ('confirm', 'support', 'show', 'demonstrate', 'agree',
		'true', 'valid', 'report', 'fact', 'factual', 'prove', 'proven',
		'evidence', 'confirmed', 'truth', 'right', 'correct', 'concur')

hypothesis_words = ('argue', 'argument', 'believe', 'belief', 'conjecture',
      'consider', 'hint', 'hypothesis', 'hypotheses', 'hypothesize', 'implication',
      'imply', 'indicate', 'predict', 'prediction', 'previous', 'previously',
      'proposal', 'propose', 'question', 'speculate', 'speculation', 'suggest',
      'suspect', 'theorize', 'theory', 'think', 'whether')

neg_words = ('not', 'no', 'none', 'nothing', 'nobody', 'never')
	

def pnow():
	print(dtime.now())

def remove_stopwords(article, stopwords = stopwords.words('english') + ['ha', 'wa']):
	return [w for w in article if w not in stopwords]

def get_antonyms(word):
	antos = set()
	for s in wn.synsets(word):
		for l in s.lemmas():
			for a in l.antonyms():
				antos.add(a.name())
	return antos

def get_syns_of_pos(word, pos):
	if pos == 'v':
		s_set = wn.synsets(word, pos=wn.VERB)
	elif pos == 'n':
		s_set = wn.synsets(word, pos=wn.NOUN)

	## remove the POS info from the WN term (probably a better way to do this than regex...)
	#syns =  set([re.sub(r'\..*$','', w.name().lower()) for w in syns])
	synonyms = set()
	for s in s_set:
		for l in s.lemmas():
			synonyms.add(l.name())
	# now, get rid of anything that has a _ in it
	synonyms = [w for w in list(synonyms) if not re.search(r'_',w)]
	return synonyms

#####################################3

def pm_syntax(targets, article, rel_words, rel_sufs = None, r = 5):
	# "poor man's" syntax
	# go through the list of article terms and see if we see any of the target terms
	# then, go through and see if there are any of the relevant words in the **scope**
	# basically, within X number of words

	# crude but might just work.....
	
	relevant_prox_freq = 0
	for i, l in enumerate(article):
		if l not in targets:
			continue
		else:
			# when we see one of the terms, go back 'neg_range' and forward 'neg_range'
			# if there's a negative word, add 1 to our count
			# 
			for j in range(max(0, i-r),min(len(article), i+r)):
				if article[j] in rel_words:
					relevant_prox_freq += 1
				if rel_sufs != None and re.search(rel_sufs, article[j]):
					relevant_prox_freq += 1

	return relevant_prox_freq

def pm_negation(targets, article, r = 3):
	# how many times we've seen a negation word around the target terms
	neg_prox_freq = pm_syntax(targets, article, neg_words, r'n\'t$', r = r)
					
	return neg_prox_freq

def pm_refute(targets, article, r = 6):
	refute_prox_freq = pm_syntax(targets, article, refuting_words, rel_sufs = None, r=r)
	return refute_prox_freq

def pm_agree(targets, article, r = 6):
	agree_prox_freq = pm_syntax(targets, article, agree_words, rel_sufs = None, r=r)
	return agree_prox_freq

def pm_hypothesis(targets, article, r = 6):
	hypothesis_prox_freq = pm_syntax(targets, article, hypothesis_words, rel_sufs = None, r=r)
	return hypothesis_prox_freq

def pm_antonyms(targets, article, r = 10):
	# look to see if any antonyms are in proximity
	antonyms = set()
	for t in targets:
		antonyms = antonyms.union(get_antonyms(t))
	antonyms = list(antonyms)
	if len(antonyms) > 0:
		anto_prox_freq = pm_syntax(targets, article, antonyms, rel_sufs = None, r=r)
	else:
		anto_prox_freq = 0
	return anto_prox_freq

###############################################
def count_negation(article):
	neg_count = 0
	for w in article:
		if w in neg_words or re.search(r'n\'t$', w):
			neg_count+=1
	return neg_count

def count_refute(article):
	refute_count = 0
	for w in article:
		if w in refuting_words:
			refute_count += 1
	return refute_count

def count_agree(article):
	agree_count = 0
	for w in article:
		if w in agree_words:
			agree_count += 1
	return agree_count

def count_hypothesis(article):
	hypothesis_count = 0
	for w in article:
		if w in hypothesis_words:
			hypothesis_count += 1

	return hypothesis_count

###############################################

def shared_ngrams(head, body, n=2):
	head_ngrams = ngrams(head, n)
	body_ngrams = ngrams(body, n)
	shared = 0
	for h in head_ngrams:
		if h in body_ngrams:
			shared += 1
	return shared

################################################
def get_dists(targets, article):
	# go through an article body, and calculate the avg distance between
	# target words in the document

	last_index = -1
	dists = [0]
	for i, w in enumerate(article):
		if w in targets:
			if last_index > 0:
				dists.append(i-last_index)
				last_index = i
			else:
				last_index = i
	max_dist = max(dists)
	avg_dist = sum(dists)/max(len(dists),1)
	return max_dist, avg_dist 
###################################################
def find_antonyms(targets, article):
	# first, find all ANTONYMS of the passed words
	# then, count how many show up in the article
	antonym_count = 0

	# get the antonyms of all target words
	antos = set()
	for t in targets:
		antos = antos.union(get_antonyms(t))
		
	for i, w in enumerate(article):
		if w in antos:
			antonym_count += 1

	return antonym_count
###################################################
###################################################
def add_features(headline_terms, body_terms, headline, 
	stemmer = None, lemmatizer = None):
	f_dict = {}

	# get all the NOUNS and VERBS from the headline
	# motivation:
	# to tell whether or not a document agrees/disagrees/discusses
	# the topic of a head line, we'll look for the actions (verbs)
	# and the subject/object of those verbs (nouns) in the head line

	# we THEN go through the document and see how many of those relevant
	# terms from the headline are proximate to any words that usually encode
	# disagreement, agreement, discussion, negation, etc.

	# Grad Student BONUS!
	# we can ALSO use WordNet to both find the synonyms of the important terms
	# AND th antonyms and look around for those
	head_nouns = []
	head_verbs = []
	for token, pos in pos_tag(word_tokenize(headline)):
		if re.search(r'^NN', pos):
			if stemmer != None:
				head_nouns.append(stemmer.stem(token))
			elif lemmatizer != None:
				head_nouns.append(lemmatizer.lemmatize(token))
			else:
				head_nouns.append(token)
		elif re.search(r'^VB', pos):
			if stemmer != None:
				head_verbs.appen(stemmer.stem(token))
			elif lemmatizer != None:
				head_verbs.append(lemmatizer.lemmatize(token))
			else:		
				head_verbs.append(token)
				

	# find the synonyms of the verbs in the headline 
	# and add them to our list
	verbs_to_add = []
	for v in head_verbs:
		verbs_to_add.extend(get_syns_of_pos(v, 'v'))
	head_verbs = list(set(head_verbs + verbs_to_add))
	
	# find synonyms for nouns in the headline....
	nouns_to_add = []
	for n in head_nouns:
		nouns_to_add.extend(get_syns_of_pos(n, 'n'))
	head_nouns = list(set(head_nouns + nouns_to_add))

	# add the features!
	f_dict['head_neg_v'] = pm_negation(head_verbs, body_terms)
	f_dict['body_neg_n'] = pm_negation(head_nouns, body_terms)
	
	f_dict['head_ref_v'] = pm_refute(head_verbs, body_terms)
	f_dict['head_ref_n'] = pm_refute(head_nouns, body_terms)
	
	f_dict['head_agr_v'] = pm_agree(head_verbs, body_terms)
	f_dict['head_agr_n'] = pm_agree(head_nouns, body_terms)

	f_dict['head_hypothesis_v'] = pm_hypothesis(head_verbs, body_terms)
	f_dict['head_hypothesis_n'] = pm_hypothesis(head_nouns, body_terms)

	f_dict['antos_v'] = pm_antonyms(head_verbs, body_terms)
	f_dict['antos_n'] = pm_antonyms(head_nouns, body_terms)

	f_dict['neg_ref'] = pm_negation(refuting_words, body_terms, r=1)
	f_dict['neg_agr'] = pm_negation(agree_words, body_terms, r=1)
	f_dict['neg_hypothesis'] = pm_negation(hypothesis_words, body_terms, r=1)

	f_dict['shared_2'] = shared_ngrams(headline_terms, body_terms,n = 2)
	f_dict['shared_3'] = shared_ngrams(headline_terms, body_terms,n = 3)
	f_dict['shared_4'] = shared_ngrams(headline_terms, body_terms,n = 4)

	f_dict['counts_antos_v'] = find_antonyms(head_verbs, body_terms)
	f_dict['counts_antos_n'] = find_antonyms(head_nouns, body_terms)

	f_dict['counts_head_neg'] = count_negation(headline_terms)
	f_dict['counts_head_ref'] = count_refute(headline_terms)
	f_dict['counts_head_agr'] = count_agree(headline_terms)
	f_dict['counts_head_hypothesis'] = count_hypothesis(headline_terms)
	
	f_dict['counts_body_neg'] = count_negation(body_terms)
	f_dict['counts_body_ref'] = count_refute(body_terms)
	f_dict['counts_body_agr'] = count_agree(body_terms)
	f_dict['counts_body_hypothesis'] = count_hypothesis(body_terms)
	
	f_dict['dist_v_max'], f_dict['dist_v_avg'] = get_dists(head_verbs, body_terms)
	f_dict['dist_n_max'], f_dict['dist_n_avg'] = get_dists(head_nouns, body_terms)
	f_dict['dist_all_max'], f_dict['dist_all_max'] = get_dists(head_verbs + head_nouns, body_terms)

	return f_dict
####

def build_training(bodies, stances, article_lems, head_lems, 
	do_stem = False, do_lemmatize = True, 
	class_ignore = ['unrelated'], class_merge = None):
	rowIDs = []
	Xs = []
	ys = []
	# if we want to do stemming....
	if do_stem:
		stemmer = SnowballStemmer("english", ignore_stopwords=True)
		lemmatizer = None
	# if we want to lemmatize...
	# note: we only lemmatize OR stem, not both (and prefer to stem)
	elif do_lemmatize:
		stemmer = None
		lemmatizer = WordNetLemmatizer()
	# if we're doing nothing
	else:
		stemmer = None
		lemmatizer = None

	# we're only doing the SVM for certain classes, filter the passed
	# pandas df to ONLY have stances of certain values
	# make a string for the query that will stop unwanted stances from
	# sneaking in...
	if len(class_ignore) == 1:
		q_string = 'Stance != \"' + class_ignore[0] + '\"'
	else:
		q_string = ' and '.join(['Stance != \"' + s + '\"' for s in class_ignore])
	
	related_stances =  stances.query(q_string)
	
	for i, row in stances.iterrows():
	# get the headline text, body text and stance
		headline = row['Headline'].lower()
		body = bodies.loc[bodies['Body ID']== row['Body ID']].iloc[0,1]
		stance = row['Stance']

		# if we're merging classes, to that here
		# e.g. we want to treat "agree" and "discuss" as the same class
		# to distinguish from "disagree"
		if class_merge != None and stance in class_merge:
			stance = class_merge[stance]

		# if it's a class we want to ignote, skip
		if stance in class_ignore:
			continue
	# get the index in the list of terms for the articles for the body
		body_index = bodies.loc[bodies['Body ID']== row['Body ID']].index[0]
		body_terms = remove_stopwords(article_lems[body_index])

	# get the headline terms
		headline_terms = remove_stopwords(head_lems[i])

	# create a dictionary with the proper features and their values for this
	# headline/document pair
		feature_dict = add_features(headline_terms, body_terms, headline,
			stemmer = stemmer, lemmatizer = lemmatizer)
	
	# add the features to our running list of docs and their features
		little_X = []
		for f in sorted(feature_dict):
			little_X += [feature_dict[f]]
	
	# add the featurized value and the label to our list
		Xs.append(little_X)
		ys.append(stance)
	# also get the row labe for this prediction
		rowIDs.append(i)
		print(len(rowIDs), end = '\r')

		if len(Xs) > 1000:
			#break
			pass
	print()
	# convert the Xs to numpy and then return
	return numpy.asarray(Xs, dtype='float16'), ys, rowIDs
####

def split_train_test(Xs, ys, test_split = .8):
	test_split = int(Xs.shape[0] * .8)
	train_X = Xs[:test_split]
	train_y = ys[:test_split]
	test_X = Xs[test_split:]
	test_y = ys[test_split:]
	return (train_X, train_y), (test_X, test_y)


#####################################################################
