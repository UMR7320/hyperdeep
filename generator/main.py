#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 20 dec. 2020
@author: lauren

t.vanni@unice.fr
'''
import numpy as np
from keras.utils import np_utils

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
	preprocessing = Filtering()
	preprocessing.loadData(corpus_file, config)

	# Train Language model
	language = Language()
	model = language.getModel(config, input_size=config["WORD_LENGTH"], output_size=len(preprocessing.unique_words))
	history = model.fit(preprocessing.X, preprocessing.Y, validation_split=0.05, epochs=10, batch_size=32, shuffle=True)

# ------------------------------
# GENERATE
# ------------------------------
def generate(model_file, text_file):
	pass