#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 20 dec. 2020
@author: lauren

t.vanni@unice.fr
'''
import random
import os
import json
import math
import numpy as np
from keras.utils import np_utils
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model

from preprocess.filtering import Filtering
from generator.language import Language

import nltk

config = {}

config["WORD_LENGTH"] = 10
config["EMBEDDING_SIZE"] = 300
config["LSTM_SIZE"] = 256
config["LEARNING_RATE"] = 0.001
config["DENSE_LAYER_SIZE"] = 1000
config["DROPOUT_VAL"] = 0.2

def z_score(k, f, t, T):
	# probability to find the word in the corpus
	p = f/float(T)
	# expected number of occurrences of the word in the text
	mean = t*p
	# standard deviation
	sdev = math.sqrt(t*p*(1-p))

	# z-score calculation
	return (k-mean)/sdev

# ------------------------------
# TRAIN
# ------------------------------
def train(corpus_file, model_file):

	# preprocess data
	preprocessing = Filtering(model_file)
	preprocessing.loadData(corpus_file, config)

	# Train Language model
	language = Language()
	model = language.getModel(config, input_size=config["WORD_LENGTH"], output_size=len(preprocessing.unique_words))

	checkpoint = ModelCheckpoint(model_file+".lang", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	earlystop = EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=3, verbose=1, mode='max')
	callbacks_list = [checkpoint, earlystop]
	history = model.fit(preprocessing.X, preprocessing.Y, validation_split=0.05, epochs=10, batch_size=1024, shuffle=True, callbacks=callbacks_list)

# ------------------------------
# GENERATE
# ------------------------------
def generate(model_file, text_file):
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = ""
	
	corpus_file = model_file.replace("bin/", "data/" )
	preprocessing = Filtering(model_file)
	preprocessing.loadData(corpus_file, config)

	# Get spec
	spec = json.loads(open(model_file + ".spec", "r").read())
	vocab = list(spec.keys())
	myNgrams = list(nltk.ngrams(preprocessing.text, 3))
	#freq_bi = nltk.FreqDist(myBigrams)
	#print(freq_bi.most_common(20))

	model = load_model(model_file+".lang")
	text = ""
	concate = []
	for i in range(100):

		# Get next predicted word
		preprocessing.loadTest(text_file, config, concate)
		predictions = list(model.predict(preprocessing.X)[0])
		max_pred = predictions.index(max(predictions))
		#weighted_predictions = [p*10000 for p in predictions]
		#max_pred = random.choices(range(len(weighted_predictions)), weights=weighted_predictions, k=1)[0]
		prediction = preprocessing.unique_index_word[max_pred]
		
		#if len(concate) <= 5:
		#	print(prediction)
	    
	    # ----------------------------------------------------		
		# Adjust prediction
		if len(concate) > config["WORD_LENGTH"]:

			"""
			try:
				pred_count = concate.count(prediction)+1
				k = spec[prediction]["k"] + pred_count
				f = spec[prediction]["f"] + pred_count
				t = spec[vocab[0]]["t"] + len(concate)+1
				T = spec[vocab[0]]["T"] + len(concate)+1
				z1 = z_score(spec[prediction]["k"], spec[prediction]["f"], spec[vocab[0]]["t"], spec[vocab[0]]["T"])
				z2 = z_score(k, f, t, T)
				delta = z1 - z2
				#print(prediction, k, f, t, T, z1, z2)
			except:
				delta = 0

			#if delta != 0:
				print(prediction, delta)
			"""

			try:
				while (concate[-2], concate[-1], prediction) not in myNgrams or prediction == concate[-1]:
					
					#predictions[max_pred] = -1
					max_pred = predictions.index(max(predictions))	
					#max_pred = random.choices(range(len(weighted_predictions)), weights=weighted_predictions, k=1)[0]
					prediction = preprocessing.unique_index_word[max_pred]

					"""
					try:
						pred_count = concate.count(prediction)+1
						k = spec[prediction]["k"] + pred_count
						f = spec[prediction]["f"] + pred_count
						t = spec[vocab[0]]["t"] + len(concate)+1
						T = spec[vocab[0]]["T"] + len(concate)+1
						z1 = z_score(spec[prediction]["k"], spec[prediction]["f"], spec[vocab[0]]["t"], spec[vocab[0]]["T"])
						z2 = z_score(k, f, t, T)
						delta = z1 - z2
					except:
						delta = 0
					"""
					# print("\t", prediction, delta)
			except:
				pass

		# ----------------------------------------------------
		concate += [prediction]
		print(concate)
		
		if concate[-(config["WORD_LENGTH"]-1):] == concate[-(config["WORD_LENGTH"]-1)*2:-(config["WORD_LENGTH"]-1)]:
			print("repeat:", concate[-(config["WORD_LENGTH"]-1):])
			concate = concate[:-(config["WORD_LENGTH"]-1)]
			for j in range(len(preprocessing.text)):
				if preprocessing.text[j:j+3] == concate[-3:]:
					print("replace with:", preprocessing.text[j+3:j+3+config["WORD_LENGTH"]])
					concate += preprocessing.text[j+3:j+3+config["WORD_LENGTH"]]
					print(concate)
					break

	print("-"*50)
	print(preprocessing.X_test.strip() + " " + " ".join(concate))