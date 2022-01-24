#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 16 nov. 2017
@author: laurent.vanni@unice.fr
'''
import time
import numpy as np
import random
import os
import pickle
import re
import os
import multiprocessing
from collections import Counter

from nltk import ngrams
from nltk import FreqDist

from gensim.models import Word2Vec

print("IMPORT SPACY")
import spacy

# ----------------------------------------
# Filter datas
# ----------------------------------------
class PreProcessing:

	def __init__(self, model_name):
		self.model_name = model_name
		print("SPACY LOAD")
		self.nlp = spacy.load("fr_core_news_lg", exclude=["ner", "parser"])
		try:
			self.dictionary = {}
			self.indexes = {}
			self.sizeOfdictionary = {}
			self.sgram = {}
			self.lgram = {}

			for wtype in ["FORME", "CODE"]:
				with open(model_name + "_" + wtype + ".index", 'rb') as handle:
					print("OPEN EXISTING DICTIONARY:", model_name + ".index")
					self.dictionary[wtype] = pickle.load(handle)
					self.sizeOfdictionary[wtype] = len(self.dictionary.keys())
					self.indexes[wtype] = {v: k for k, v in self.dictionary[wtype].items()}

				with open(model_name + "_" + wtype + ".sgram", 'rb') as handle:
					print("OPEN EXISTING SGRAM:", model_name + ".index")
					self.sgram[wtype] = pickle.load(handle)

				with open(model_name + "_" + wtype + ".lgram", 'rb') as handle:
					print("OPEN EXISTING LGRAM:", model_name + ".index")
					self.lgram[wtype] = pickle.load(handle)
		except:
			print("NO DICTIONARY DETECTED")


	# ----------------------------------------
	# NLP : get code
	# ----------------------------------------		
	def get_code(self, token):
		code = token.tag_
		if code != "SPACE":
			for key, value in token.morph.to_dict().items():
				code += ":" + key + "_" + value
		return code


	# ----------------------------------------
	# Load data from file
	# ----------------------------------------
	def loadCorpus(self, corpus, config):  

		print("SPACY ANALYSE")
		docs = list(self.nlp.pipe(corpus, n_process=-1, batch_size=8))
		print("SPACY DONE.")

		self.text = {}
		self.text["FORME"] = []
		self.text["CODE"] = []

		self.dictionary = {}
		self.sizeOfdictionary = {}
		self.embedding_matrix = {}
		self.sgrams = {}
		self.lgrams = {}
		self.X = {}
		self.Y = {}

		for doc in docs:
			for token in doc:
				# TOKENIZE
				self.text["FORME"] += [token.text]
				self.text["CODE"] += [self.get_code(token)]

		# --------------------------------		
		# PREPROCESS
		for wtype in ["FORME", "CODE"]:
			# MOST FREQUENT WORDS
			freqDist = FreqDist(self.text[wtype])
			most_commont_list = [entry[0] for entry in freqDist.most_common(config["VOCAB_SIZE"])]
			print(most_commont_list[:10])

			print("VOCAB_SIZE", config["VOCAB_SIZE"])

			# CREATE DICTIONARY
			print("Create dictionary...")
			indexes = dict((i+1, c) for i, c in enumerate(most_commont_list))
			indexes[0] = "PAD"
			dictionary = dict((c, i) for i, c in indexes.items())
			vocabulary = list(dictionary.keys())
			sizeOfdictionary = len(vocabulary)
			print("Size of dictionary:", sizeOfdictionary)

			# FILTER CORPUS
			sentences = []
			sentence = []
			keepSentence = True
			for i, word in enumerate(self.text[wtype]):
				
				if word not in most_commont_list:
					keepSentence = False
				elif self.text["CODE"][i] != "SPACE":
					sentence += [word]
				
				if self.text["CODE"][i] == "PUNCT":
					if keepSentence:
						sentences += sentence
					else:
						keepSentence = True
					sentence = []

			WORD_LENGTH = config["WORD_LENGTH"]
			prev_words = []
			next_words = []
			for i in range(len(sentences) - WORD_LENGTH):
				prev_words.append(sentences[i:i + WORD_LENGTH])
				next_words.append(sentences[i + WORD_LENGTH])
			print("NUMBER OF SAMPLE:", len(next_words))

			# COMPUTE NGRAM
			print("COMPUTE NGRAM...")
			sgrams = list(ngrams(sentences, 3))
			lgrams = list(ngrams(sentences, WORD_LENGTH))

			# create two numpy arrays x for storing the features and y for storing its corresponding label
			X = np.zeros((len(prev_words), WORD_LENGTH), dtype=int)
			Y = np.zeros((len(next_words), sizeOfdictionary), dtype=int)
			for i, each_words in enumerate(prev_words):
				for j, each_word in enumerate(each_words):
					X[i, j] = dictionary[each_word]
				Y[i, dictionary[next_words[i]]] = 1

			print(X[:10])
			print(Y[:10])
			print("DATA LEN = ", len(X))

			# --------------------------------
			# Word2Vec
			print("process word2vec...")
			cores = multiprocessing.cpu_count() # Count the number of cores in a computer
			print(prev_words[:10])
			self.w2v_model = Word2Vec(min_count=0,
	                     window=5,
	                     size=300,
	                     sample=6e-5, 
	                     alpha=0.03, 
	                     min_alpha=0.0007, 
	                     negative=0,
	                     workers=cores-1)
			self.w2v_model.build_vocab(prev_words, progress_per=10000)
			self.w2v_model.train(prev_words, total_examples=self.w2v_model.corpus_count, epochs=10, report_delay=1)

			print("vocab size", len(self.w2v_model.wv.vocab.keys()))
			#print("France:", self.w2v_model.wv['France'])

			# Get vectors from the w2v model
			self.embedding_matrix[wtype] = np.zeros((sizeOfdictionary, config["EMBEDDING_SIZE"]))
			for i in range(sizeOfdictionary):
				try:
					embedding_vector = self.w2v_model.wv[vocabulary[i]]
					self.embedding_matrix[wtype][i] = embedding_vector
				except:
					print(vocabulary[i], "not in vocabulary")
			# --------------------------------

			# Compute attributes
			self.dictionary[wtype] = dictionary
			self.sizeOfdictionary[wtype] = sizeOfdictionary
			self.sgrams[wtype] = sgrams
			self.lgrams[wtype] = lgrams
			self.X[wtype] = X
			self.Y[wtype] = Y
			# Store datas
			pickle.dump(dictionary, open(self.model_name + "_" + wtype + ".index", "wb"))
			pickle.dump(sgrams, open(self.model_name + "_" + wtype + ".sgram", "wb"))
			pickle.dump(lgrams, open(self.model_name + "_" + wtype + ".lgram", "wb"))

	# ----------------------------------------
	# Load bootstrap from file
	# ----------------------------------------
	def loadSequence(self, bootstrap, config, concate={"FORME":[], "CODE":[]}):   

		self.X = {}
		for i, wtype in enumerate(["FORME", "CODE"]):
		#for i, wtype in enumerate(["FORME"]):

			# Load index
			WORD_LENGTH = config["WORD_LENGTH"]

			# compute X_bootstrap
			self.X[wtype] = np.zeros((1, WORD_LENGTH), dtype=int)
			text = bootstrap[wtype] + concate[wtype]
			text = text[-WORD_LENGTH:]
			for j, each_word in enumerate(text):
				try:
					self.X[wtype][0, j] = self.dictionary[wtype][each_word]
				except:
					self.X[wtype][0, j] = self.dictionary[wtype]["PAD"]
				#print(each_word + "_" + str(self.X[wtype][0, j]) + " ", end=' ', flush=True)
			#print()