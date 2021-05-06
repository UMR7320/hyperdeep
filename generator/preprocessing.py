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

from nltk import ngrams
from nltk import FreqDist

print("IMPORT SPACY")
import spacy

# ----------------------------------------
# Filter datas
# ----------------------------------------
class PreProcessing:

	def __init__(self, model_name):
		self.model_name = model_name
		print("SPACY LOAD")
		self.nlp = spacy.load("fr_core_news_sm", exclude=["ner", "parser"])
		try:
			self.dictionary = {}
			self.sizeOfdictionary = {}
			self.sgram = {}
			self.lgram = {}

			for wtype in ["FORME", "CODE"]:
				with open(model_name + "_" + wtype + ".index", 'rb') as handle:
					print("OPEN EXISTING DICTIONARY:", model_name + ".index")
					self.dictionary[wtype] = pickle.load(handle)
					self.sizeOfdictionary[wtype] = len(self.dictionary.keys())

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
	def loadCorpus(self, corpus_file, config):   

		f = open(corpus_file, "r")
		lines = f.readlines()

		print("SPACY ANALYSE")
		docs = list(self.nlp.pipe(lines, n_process=-1, batch_size=8))
		print("SPACY DONE.")

		self.text = {}
		self.text["FORME"] = []
		self.text["CODE"] = []

		for doc in docs:
			for token in doc:
				# TOKENIZE
				self.text["FORME"] += [token.text]
				self.text["CODE"] += [self.get_code(token)]

		"""
		# --------------------------------
		# Word2Vec
		# HYPERPARAMETERS
		EMBEDDING_SIZE = 300
		MIN_COUNT = 0
		WINDOW_SIZE = 10

		print("process word2vec...")
		self.w2v_model = Word2Vec(min_count=MIN_COUNT, window=WINDOW_SIZE, size=config["EMBEDDING_SIZE"], sg=1, workers=8)
		self.w2v_model.build_vocab([self.text], progress_per=10000)
		self.w2v_model.train(self.text, total_examples=self.w2v_model.corpus_count, epochs=10, report_delay=1)

		# Get vectors from the w2v model
		self.vectors = self.w2v_model.wv
		#print(list(vectors.vocab)[:10])
		# --------------------------------
		"""

		self.dictionary = {}
		self.sizeOfdictionary = {}
		self.sgrams = {}
		self.lgrams = {}
		self.X = {}
		self.Y = {}

		# --------------------------------
		# Keep only most frequent words
		for wtype in ["FORME", "CODE"]:

			text = self.text[wtype]

			freqDist = FreqDist(text)
			most_commont_list = [entry[0] for entry in freqDist.most_common(config["VOCAB_SIZE"])]
			#self.text = [word for word in self.text if word in most_commont_list]

			# Create a new dictionary
			dictionary = dict((c, i) for i, c in enumerate(most_commont_list))
			sizeOfdictionary = len(dictionary.keys())
			pickle.dump(dictionary, open(self.model_name + "_" + wtype + ".index", "wb"))

			# FILTER CORPUS
			WORD_LENGTH = config["WORD_LENGTH"]
			prev_words = []
			next_words = []
			for i in range(len(text) - WORD_LENGTH):
				if any(w not in most_commont_list for w in text[i:i + WORD_LENGTH + 1]) : continue
				if text[i:i + WORD_LENGTH + 1].count("\n") > 1 : continue
				prev_words.append(text[i:i + WORD_LENGTH])
				next_words.append(text[i + WORD_LENGTH])
			print("NUMBER OF SAMPLE:", len(next_words))
			print(prev_words[:10])

			# COMPUTE NGRAM
			sgrams = list(ngrams(text, 3))
			pickle.dump(sgrams, open(self.model_name + "_" + wtype + ".sgram", "wb"))
			lgrams = list(ngrams(text, config["WORD_LENGTH"]))
			pickle.dump(lgrams, open(self.model_name + "_" + wtype + ".lgram", "wb"))

			# create two numpy arrays x for storing the features and y for storing its corresponding label
			X = np.zeros((len(prev_words), WORD_LENGTH, sizeOfdictionary), dtype=bool)
			Y = np.zeros((len(next_words), sizeOfdictionary), dtype=bool)
			for i, each_words in enumerate(prev_words):
				for j, each_word in enumerate(each_words):
					X[i, j, dictionary[each_word]] = 1
				Y[i, dictionary[next_words[i]]] = 1

			self.dictionary[wtype] = dictionary
			self.sizeOfdictionary[wtype] = sizeOfdictionary
			self.sgrams[wtype] = sgrams
			self.lgrams[wtype] = lgrams
			self.X[wtype] = X
			self.Y[wtype] = Y

	# ----------------------------------------
	# Load bootstrap from file
	# ----------------------------------------
	def loadSequence(self, bootstrap, config, concate=[]):   

		self.X = {}
		self.reversedictionary = {}
		for i, wtype in enumerate(["FORME", "CODE"]):

			self.reversedictionary[wtype] = {v: k for k, v in self.dictionary[wtype].items()}

			# Load index
			WORD_LENGTH = config["WORD_LENGTH"]

			# compute X_bootstrap
			self.X[wtype] = np.zeros((1, WORD_LENGTH, len(self.dictionary[wtype].keys())), dtype=bool)
			text = bootstrap[wtype] + concate #[c[i] for c in concate]
			text = text[-WORD_LENGTH:]
			for j, each_word in enumerate(text):
				try:
					self.X[wtype][0, j, self.dictionary[wtype][each_word]] = 1
				except:
					self.X[wtype][0, j, 0] = 1