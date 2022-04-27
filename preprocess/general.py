#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 27 apr. 2022
@author: laurent.vanni@unice.fr
'''
import time
import numpy as np
import random
import os
import pickle
import re

from keras.utils import np_utils
from preprocess.w2vec import create_vectors

# ----------------------------------------
# Preprocess text from input file
# Format : 
# __LABEL1__ word1 word2 word3 ...
# __LABEL2__ word1 word2 word3 ...
# ----------------------------------------
class PreProcessing:

	# ------------------------------
	# INIT
	# ------------------------------
	def __init__(self, model_file, config):
		self.model_file = model_file
		self.config = config
		self.num_classes = len(config["CLASSES"])

	# ----------------------------------------
	# Load data from file
	# ----------------------------------------
	def loadData(self, corpus_file):   
		
		print("loading data...")
		print("NB CHANNELS:", self.config["nb_channels"])
		
		self.corpus_file = corpus_file
		self.raw_texts = []
		self.channel_texts = {}
		self.labels = []

		f = open(corpus_file, "r")
		lines = f.readlines()

		cpt = 0
		t0 = time.time()
		print("-"*50)
		print("PREPROCESS SAMPLES")
		print("-"*50)
		for line in lines:
			if "--" in line: continue

			if cpt%100 == 0:
				t1 = time.time()
				print("sample", cpt, "/", len(lines))
				t0 = t1

			# LABELS
			if line[:2] == "__" and line[2:8] != "PARA__":
				try:
					label = line.split("__ ")[0].replace("__", "")
					label_int = self.config["CLASSES"].index(label)
					self.labels += [label_int]
					line = line.replace("__" + label + "__ ", "")
				except:
					raise
					print("error with line:", line)

			self.raw_texts += [line]

			# TEXT
			sequence = []
			for i in range(self.config["nb_channels"]):
				sequence += [[]]

			for token in line.split():
				args = token.split("**")
				for i in range(self.config["nb_channels"]):
					try:
						if not args[i]:
							sequence[i] += [PAD]
						else:
							sequence[i] += [args[i]]
					except:
						sequence[i] += "PAD "

			for i in range(self.config["nb_channels"]):
				self.channel_texts[i] = self.channel_texts.get(i, [])
				self.channel_texts[i].append(sequence[i])
		
			cpt += 1
		f.close()
	
	# ---------------------------------------------
	# Encode data : Convert data to numerical array
	# forTraining=True : 
	#		- create a new dictionary
	#		- Split data (Train/Validation)
	# ---------------------------------------------
	def encodeData(self, forTraining=False):

		dictionaries, datas = self.loadIndex(forTraining)

		for i, dictionary in enumerate(dictionaries):
			print('Found %s unique tokens in channel ' % len(dictionary["word_index"]), i+1)

		# Size of each dataset (train, valid, test)
		nb_validation_samples = int(self.config["VALIDATION_SPLIT"] * datas[0].shape[0])
		nb_testing_samples = nb_validation_samples + int(self.config["TESTING_SPLIT"] * datas[0].shape[0])

		# split the data into a training set and a validation set
		indices = np.arange(datas[0].shape[0])		
		if forTraining:
			np.random.shuffle(indices)
			self.labels = np_utils.to_categorical(np.asarray(self.labels))
			print('Shape of label tensor:', self.labels.shape)
			self.labels = self.labels[indices]
			self.y_val = self.labels[:nb_validation_samples]
			self.y_test = self.labels[nb_validation_samples:nb_testing_samples]
			self.y_train = self.labels[nb_testing_samples:]

		self.x_train = []
		self.x_val = []
		self.x_test = []
		self.x_data = []
		for data in datas:
			data = data[indices]
			self.x_val += [data[:nb_validation_samples]]
			self.x_test += [data[nb_validation_samples:nb_testing_samples]]
			self.x_train += [data[nb_testing_samples:]]

		self.dictionaries = dictionaries

	# ----------------------------------------
	# Train Word2Vec embeddings
	# ----------------------------------------
	def loadEmbeddings(self):

		print("CREATE WORD2VEC VECTORS")
		create_vectors(self.channel_texts, self.model_file, self.config)

		# Make embedding_matrix from vectors
		self.embedding_matrix = []
		for i in range(self.config["nb_channels"]):
			my_dictionary = self.dictionaries[i]["word_index"]
			embeddings_index = {}
			vectors = open(self.model_file + ".word2vec" + str(i) ,'r')
			
			for line in vectors.readlines():
				values = line.split()
				word = values[0]
				coefs = np.asarray(values[1:], dtype='float32')
				embeddings_index[word] = coefs

			print('Found %s word vectors.' % len(embeddings_index))
			self.embedding_matrix += [np.zeros((len(my_dictionary), self.config["EMBEDDING_DIM"]))]
			for word, j in my_dictionary.items():
				embedding_vector = embeddings_index.get(word)
				if embedding_vector is not None:
					# words not found in embedding index will be all-zeros.
					self.embedding_matrix[i][j] = embedding_vector
			vectors.close()

	# ----------------------------------------
	# LoadIndex
	# CREATE WORD INDEX DICTIONARY
	# ----------------------------------------
	def loadIndex(self, forTraining):

		if forTraining:
			print("CREATE A NEW DICTIONARY")
			dictionaries = []
			indexes = [1,1,1]
			for i in range(3):
				dictionary = {}
				dictionary["word_index"] = {}
				dictionary["index_word"] = {}
				dictionary["word_index"]["PAD"] = 0  # Padding
				dictionary["index_word"][0] = "PAD"
				dictionary["word_index"]["__UK__"] = 1 # Unknown word
				dictionary["index_word"][1] = "__UK__" 
				dictionaries += [dictionary]
		else:
			with open(self.model_file + ".index", 'rb') as handle:
				print("OPEN EXISTING DICTIONARY:", self.model_file + ".index")
				dictionaries = pickle.load(handle)
		datas = []		

		for channel, text in self.channel_texts.items():
			datas += [(np.zeros((len(text), self.config["SEQUENCE_SIZE"]))).astype('int32')]	

			line_number = 0
			for i, line in enumerate(text):
				
				words = line[:self.config["SEQUENCE_SIZE"]]
				sentence_length = len(words)
				sentence = []

				for j, word in enumerate(words):
					if word not in dictionaries[channel]["word_index"].keys():
						if forTraining:
							# ----------------------------
							# SKIP WORDS
							skip_word = False
							for f in self.config["FILTERS"]:
								if not f.strip(): continue
								if any(re.match(f, self.channel_texts[t][i][:self.config["SEQUENCE_SIZE"]][j]) for t in range(config["nb_channels"])):
									skip_word = True
									break
							if skip_word: 
								dictionaries[channel]["word_index"][word] = dictionary["word_index"]["__UK__"]
							# ----------------------------

							else:	 
								indexes[channel] += 1
								dictionaries[channel]["word_index"][word] = indexes[channel]
								dictionaries[channel]["index_word"][indexes[channel]] = word

						else:        
							# FOR UNKNOWN WORDS
							dictionaries[channel]["word_index"][word] = dictionaries[channel]["word_index"]["__UK__"]

					sentence.append(dictionaries[channel]["word_index"][word])

				# COMPLETE WITH PAD IF LENGTH IS < SEQUENCE_SIZE
				if sentence_length < self.config["SEQUENCE_SIZE"]:
					for j in range(self.config["SEQUENCE_SIZE"] - sentence_length):
						sentence.append(dictionaries[channel]["word_index"]["PAD"])
				
				datas[channel][line_number] = sentence
				line_number += 1

		if forTraining:
			with open(self.model_file + ".index", 'wb') as handle:
				pickle.dump(dictionaries, handle, protocol=pickle.HIGHEST_PROTOCOL)

		print("VOCABULARY SIZE:", len(dictionaries[0]["index_word"]))

		return dictionaries, datas
