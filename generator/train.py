#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
Created on 20 dec. 2020
@author: lauren

t.vanni@unice.fr
'''
import os
import json
import math
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from generator.preprocessing import PreProcessing
from generator.model import Language

#import nltk
import pickle

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
def train_model(corpus_file, model_file, config):

	# preprocess data
	preprocessing = PreProcessing(model_file)
	preprocessing.loadCorpus(corpus_file, config)

	# Train Language model
	language = Language()
	model = language.getModel(config, input_size=config["WORD_LENGTH"], output_size=preprocessing.sizeOfdictionary)

	checkpoint = ModelCheckpoint(model_file+".lang", monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	earlystop = EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=3, verbose=1, mode='max')
	callbacks_list = [checkpoint, earlystop]
	history = model.fit(preprocessing.X, preprocessing.Y, validation_split=config["VALIDATION_SPLIT"], epochs=config["NUM_EPOCHS"], batch_size=config["BACH_SIZE"], shuffle=True, callbacks=callbacks_list)

