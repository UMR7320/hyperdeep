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

from nltk import FreqDist
from gensim.models import Word2Vec
from keras.utils import np_utils
from keras.preprocessing.text import hashing_trick

# ----------------------------------------
# Filter datas
# ----------------------------------------
class Filtering:

	def __init__(self, model_file):
		self.model_file = model_file

	# ----------------------------------------
	# Load data from file
	# ----------------------------------------
	def loadData(self, corpus_file, config):   

		f = open(corpus_file, "r")
		lines = f.readlines()

		self.text = [] 
		# --------------------------------
		# LOG
		cpt = 0
		t0 = time.time()
		# --------------------------------

		for line in lines:
			# --------------------------------
			# LOG
			if cpt%1000 == 0:
				t1 = time.time()
				#print("words:", cpt, "/", len(lines))
				t0 = t1
			cpt += 1
			# --------------------------------
			# FILTERING
			args = line.strip().split("\t")
			try:
				#if any(s in args[1] for s in ["NOM", "NAM", "ADJ", "VER"]) and "<unknown>" not in args[2]:
				#self.text += [args[0].strip()]

				if args[0] == "__PARA__":
					self.text += ["\n"]
				else:
					self.text += [args[0]]

			except:
				pass
			# --------------------------------

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

		# Keep only most frequent words
		"""
		freqDist = FreqDist(self.text)
		most_commont_list = [entry[0] for entry in freqDist.most_common(3000)]
		self.text = [word for word in self.text if word in most_commont_list]
		"""
		# Get word index

		self.unique_words = np.unique(self.text)
		self.unique_word_index = dict((c, i) for i, c in enumerate(self.unique_words))
		pickle.dump(self.unique_word_index, open(self.model_file + ".index", "wb"))

		# Feature Engineering
		WORD_LENGTH = config["WORD_LENGTH"]
		prev_words = []
		next_words = []
		for i in range(len(self.text) - WORD_LENGTH):
			prev_words.append(self.text[i:i + WORD_LENGTH])
			next_words.append(self.text[i + WORD_LENGTH])

		# create two numpy arrays x for storing the features and y for storing its corresponding label
		self.X = np.zeros((len(prev_words), WORD_LENGTH, len(self.unique_words)), dtype=bool)
		self.Y = np.zeros((len(next_words), len(self.unique_words)), dtype=bool)
		for i, each_words in enumerate(prev_words):
			for j, each_word in enumerate(each_words):
				self.X[i, j, self.unique_word_index[each_word]] = 1
			self.Y[i, self.unique_word_index[next_words[i]]] = 1

	# ----------------------------------------
	# Load Test from file
	# ----------------------------------------
	def loadTest(self, test_file, config, concate=[]):   

		# Load text
		f = open(test_file, "r")
		self.X_test = f.read()

		# Load index
		self.unique_word_index = pickle.load(open(self.model_file + ".index", "rb" ))
		self.unique_index_word = {v: k for k, v in self.unique_word_index.items()}
		WORD_LENGTH = config["WORD_LENGTH"]

		# compute x_test
		self.X = np.zeros((1, WORD_LENGTH, len(self.unique_word_index.keys())), dtype=bool)
		text = self.X_test.strip().split(" ") + concate
		text = text[-WORD_LENGTH:]
		for j, each_word in enumerate(text):
			try:
				self.X[0, j, self.unique_word_index[each_word]] = 1
			except:
				self.X[0, j, 0] = 1

