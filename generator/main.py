#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 20 dec. 2020
@author: lauren

t.vanni@unice.fr
'''
import os
import numpy as np
from keras.utils import np_utils
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model

from preprocess.filtering import Filtering
from generator.language import Language

config = {}

config["WORD_LENGTH"] = 5
config["EMBEDDING_SIZE"] = 300
config["LSTM_SIZE"] = 128
config["LEARNING_RATE"] = 0.01


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
	history = model.fit(preprocessing.X, preprocessing.Y, validation_split=0.05, epochs=10, batch_size=32, shuffle=True, callbacks=callbacks_list)

# ------------------------------
# GENERATE
# ------------------------------
def generate(model_file, text_file):
	model = load_model(model_file+".lang")
	preprocessing = Filtering(model_file)

	concate = []
	for i in range(20):
		preprocessing.loadTest(text_file, config, concate)
		predictions = list(model.predict(preprocessing.X)[0])
		max_pred = predictions.index(max(predictions))
		prediction = preprocessing.unique_index_word[max_pred]
		try:
			while prediction in concate:
				predictions[max_pred] = -1
				max_pred = predictions.index(max(predictions))
				prediction = preprocessing.unique_index_word[max_pred]
		except:
			pass

		print(prediction)
		concate += [prediction]
