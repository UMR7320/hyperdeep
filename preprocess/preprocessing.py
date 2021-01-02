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

from keras.utils import np_utils
from preprocess.w2vec import create_vectors

# ----------------------------------------
# Preprocess text from input file
# Format : 
# __LABEL1__ word1 word2 word3 ...
# __LABEL2__ word1 word2 word3 ...
# ----------------------------------------
class PreProcessing:

	# ----------------------------------------
	# Load data from file
	# ----------------------------------------
	def loadData(self, corpus_file, model_file, config, getLabels, createDictionary):   
		
		print("loading data...")
		
		f = open(corpus_file, "r")
		lines = f.readlines()
		self.corpus_file = corpus_file
		self.raw_text = []

		label_dic = {}
		labels = []
		texts = {}
		cpt = 0
		t0 = time.time()

		for line in lines:
			if "--" in line: continue

			if cpt%100 == 0:
				t1 = time.time()
				print(cpt, "/", len(lines))
				t0 = t1

			# LABELS
			if getLabels:
				label = line.split("__ ")[0].replace("__", "")
				label_int = config["CLASSES"].index(label)
				labels += [label_int]
				line = line.replace("__" + label + "__ ", "")

			# TEXT
			if config["TG"]:
				sequence = ["", "", ""] # MULTICHANNELS
			else:
				sequence = [""] # MONOCHANNEL
			for token in line.split():
				self.raw_text += [token]
				args = token.split("**")
				for i in range(len(sequence)):
					try:
						if not args[i]:
							sequence[i] += "PAD "
						else:
							sequence[i] += args[i] + " "
					except:
						sequence[i] += "PAD "

			for i in range(len(sequence)):
				texts[i] = texts.get(i, [])
				texts[i].append(sequence[i])
		
			cpt += 1
		f.close()

		for i, text in texts.items():
			f = open(corpus_file + "." + str(i), "w")
			for sequence in text:
				f.write(sequence + "\n")
			f.close()
		
		#print("DETECTED LABELS :")
		#print(label_dic)
		self.num_classes = len(config["CLASSES"])

		dictionaries, datas = self.tokenize(texts, model_file, createDictionary, config)

		for i, dictionary in enumerate(dictionaries):
			print('Found %s unique tokens in channel ' % len(dictionary["word_index"]), i+1)

		# Size of each dataset (train, valid, test)
		nb_validation_samples = int(config["VALIDATION_SPLIT"] * datas[0].shape[0])
		nb_testing_samples = nb_validation_samples + int(config["TESTING_SPLIT"] * datas[0].shape[0])

		# split the data into a training set and a validation set
		indices = np.arange(datas[0].shape[0])		
		if getLabels:
			np.random.shuffle(indices)
			labels = np_utils.to_categorical(np.asarray(labels))
			print('Shape of label tensor:', labels.shape)
			labels = labels[indices]
			self.y_val = labels[:nb_validation_samples]
			self.y_test = labels[nb_validation_samples:nb_testing_samples]
			self.y_train = labels[nb_testing_samples:]

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
		self.nb_channels = len(texts.keys())

	# ----------------------------------------
	# Load existing embedding
	# ----------------------------------------
	def loadEmbeddings(self, model_file, config, create_v = False):

		print("LOADING WORD2VEC EMBEDDING")
		
		self.embedding_matrix = []

		if not create_v:
			create_vectors(self.corpus_file, model_file, config, nb_channels=self.nb_channels)

		for i in range(self.nb_channels):
			my_dictionary = self.dictionaries[i]["word_index"]
			embeddings_index = {}
			vectors = open(model_file + ".word2vec" + str(i) ,'r')
				
			for line in vectors.readlines():
				values = line.split()
				word = values[0]
				coefs = np.asarray(values[1:], dtype='float32')
				embeddings_index[word] = coefs

			print('Found %s word vectors.' % len(embeddings_index))
			self.embedding_matrix += [np.zeros((len(my_dictionary), config["EMBEDDING_DIM"]))]
			for word, j in my_dictionary.items():
				embedding_vector = embeddings_index.get(word)
				if embedding_vector is not None:
					# words not found in embedding index will be all-zeros.
					self.embedding_matrix[i][j] = embedding_vector
			vectors.close()

	# ----------------------------------------
	# TOKENIZE
	# CREATE WORD INDEX DICTIONARY
	# ----------------------------------------
	def tokenize(self, texts, model_file, createDictionary, config):

		if createDictionary:
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
			with open(model_file + ".index", 'rb') as handle:
				print("OPEN EXISTING DICTIONARY:", model_file + ".index")
				dictionaries = pickle.load(handle)
		datas = []		

		type = ["FORME", "CODE", "LEM"]
		text_formes = texts[0]
		if config["TG"]:
			text_codes = texts[1]
		else:
			text_codes = False

		for channel, text in texts.items():
			datas += [(np.zeros((len(text), config["SEQUENCE_SIZE"]))).astype('int32')]	

			line_number = 0
			for i, line in enumerate(text):
				words = line.split()[:config["SEQUENCE_SIZE"]]
				
				words_formes = text_formes[i].split()[:config["SEQUENCE_SIZE"]]
				try:
					words_codes =  text_codes[i].split()[:config["SEQUENCE_SIZE"]]
				except:
					words_codes = False

				sentence_length = len(words)

				sentence = []
				for j, word in enumerate(words):
					if word not in dictionaries[channel]["word_index"].keys():
						if createDictionary:
							# IF WORD IS SKIPED THEN ADD "UK" word
							skip_word = False
							if channel != 1:
								for f in config["FILTERS"]:
									if not f.strip(): continue
									# Check Code
									skip_word = skip_word or f in words_codes[j].split(":")
									# Check Forme (regex)
									skip_word = skip_word or re.match(f, word)
									# Check Length
									if "min(" in f:
										f = f.replace("min(", "")[:-1]
										skip_word = skip_word or len(word) < int(f)
									elif "max(" in f:
										f = f.replace("max(", "")[:-1]
										skip_word = skip_word or len(word) > int(f)
									if skip_word:
										break

							if skip_word: 
								dictionaries[channel]["word_index"][word] = dictionary["word_index"]["__UK__"]
							else:	 
								indexes[channel] += 1
								dictionaries[channel]["word_index"][word] = indexes[channel]
								dictionaries[channel]["index_word"][indexes[channel]] = word

						else:        
							# FOR UNKNOWN WORDS
							dictionaries[channel]["word_index"][word] = dictionaries[channel]["word_index"]["__UK__"]

					sentence.append(dictionaries[channel]["word_index"][word])

				# COMPLETE WITH PAD IF LENGTH IS < SEQUENCE_SIZE
				if sentence_length < config["SEQUENCE_SIZE"]:
					for j in range(config["SEQUENCE_SIZE"] - sentence_length):
						sentence.append(dictionaries[channel]["word_index"]["PAD"])
				
				datas[channel][line_number] = sentence
				line_number += 1

		if createDictionary:
			with open(model_file + ".index", 'wb') as handle:
				pickle.dump(dictionaries, handle, protocol=pickle.HIGHEST_PROTOCOL)

		print("VOCABULARY SIZE:", len(dictionaries[0]["index_word"]))

		return dictionaries, datas