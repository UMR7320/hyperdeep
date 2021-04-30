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

	def __init__(self, model_file):
		self.model_file = model_file
		try:
			with open(model_file + ".index", 'rb') as handle:
				print("OPEN EXISTING DICTIONARY:", model_file + ".index")
				self.dictionary = pickle.load(handle)
				self.sizeOfdictionary = len(self.dictionary.keys())

			with open(model_file + ".sgram", 'rb') as handle:
				print("OPEN EXISTING DICTIONARY:", model_file + ".index")
				self.sgram = pickle.load(handle)

			with open(model_file + ".lgram", 'rb') as handle:
				print("OPEN EXISTING DICTIONARY:", model_file + ".index")
				self.lgram = pickle.load(handle)
		except:
			print("NO DICTIONARY DETECTED")

	# ----------------------------------------
	# Load data from file
	# ----------------------------------------
	def loadCorpus(self, corpus_file, config):   

		f = open(corpus_file, "r")
		lines = f.readlines()

		print("SPACY LOAD")
		nlp = spacy.load("fr_core_news_sm", exclude=["ner", "parser"])
		print("SPACY ANALYSE")
		docs = list(nlp.pipe(lines, n_process=-1, batch_size=8))
		print("SPACY DONE.")

		self.text = [] 
		self.pos = []

		for doc in docs:
			for token in doc:
				# TOKENIZE
				self.text += [token.text]
				self.pos += [token.tag_]

			self.text += ["\n"]
			self.pos += ["\n"]

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

		# --------------------------------
		# Keep only most frequent words
		freqDist = FreqDist(self.text)
		most_commont_list = [entry[0] for entry in freqDist.most_common(config["VOCAB_SIZE"])]
		#self.text = [word for word in self.text if word in most_commont_list]

		# Create a new dictionary
		self.dictionary = dict((c, i) for i, c in enumerate(most_commont_list))
		self.sizeOfdictionary = len(self.dictionary.keys())
		pickle.dump(self.dictionary, open(self.model_file + ".index", "wb"))

		# FILTER CORPUS
		WORD_LENGTH = config["WORD_LENGTH"]
		prev_words = []
		next_words = []
		for i in range(len(self.text) - WORD_LENGTH):
			if any(w not in most_commont_list for w in self.text[i:i + WORD_LENGTH + 1]) : continue
			if self.text[i:i + WORD_LENGTH + 1].count("\n") > 1 : continue
			prev_words.append(self.text[i:i + WORD_LENGTH])
			next_words.append(self.text[i + WORD_LENGTH])
		print("NUMBER OF SAMPLE:", len(next_words))
		print(prev_words[:10])

		# COMPUTE NGRAM
		self.sgrams = list(ngrams(self.text, 3))
		pickle.dump(self.sgrams, open(self.model_file + ".sgram", "wb"))
		self.lgrams =  list(ngrams(self.text, config["WORD_LENGTH"]))
		pickle.dump(self.lgrams, open(self.model_file + ".lgram", "wb"))

		# create two numpy arrays x for storing the features and y for storing its corresponding label
		self.X = np.zeros((len(prev_words), WORD_LENGTH, self.sizeOfdictionary), dtype=bool)
		self.Y = np.zeros((len(next_words), self.sizeOfdictionary), dtype=bool)
		for i, each_words in enumerate(prev_words):
			for j, each_word in enumerate(each_words):
				self.X[i, j, self.dictionary[each_word]] = 1
			self.Y[i, self.dictionary[next_words[i]]] = 1

	# ----------------------------------------
	# Load bootstrap from file
	# ----------------------------------------
	def loadBootstrap(self, test_file, config, concate=[]):   

		# Load text
		f = open(test_file, "r")
		self.X_bootstrap = f.read()

		# Load index
		self.dictionary = pickle.load(open(self.model_file + ".index", "rb" ))
		self.reversedictionary = {v: k for k, v in self.dictionary.items()}
		WORD_LENGTH = config["WORD_LENGTH"]

		# compute X_bootstrap
		self.X = np.zeros((1, WORD_LENGTH, len(self.dictionary.keys())), dtype=bool)
		text = self.X_bootstrap.strip().split(" ") + concate
		text = text[-WORD_LENGTH:]
		for j, each_word in enumerate(text):
			try:
				self.X[0, j, self.dictionary[each_word]] = 1
			except:
				self.X[0, j, 0] = 1

