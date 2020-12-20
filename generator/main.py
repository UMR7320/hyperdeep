#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 20 dec. 2020
@author: lauren

t.vanni@unice.fr
'''
from preprocess.preprocessing import PreProcessing

config = {}
config["TG"] = 0
config["CLASSES"] = []
config["VALIDATION_SPLIT"] = 0.2
config["TESTING_SPLIT"] = 0
config["SEQUENCE_SIZE"] = 50
config["FILTERS"] = []

# ------------------------------
# TRAIN
# ------------------------------
def train(corpus_file, model_file):

	# preprocess data
	preprocessing = PreProcessing()
	preprocessing.loadData(corpus_file, model_file, config, getLabels=False, createDictionary=True)

	# GET TRAIN DATASET
	x_train, x_val, x_test = preprocessing.x_train, preprocessing.x_val, preprocessing.x_test
	print("Available samples:")
	print("train:", len(x_train[0]), "valid:", len(x_val[0]), "test:", len(x_test[0]))


# ------------------------------
# GENERATE
# ------------------------------
def generate(model_file, text_file):
	pass