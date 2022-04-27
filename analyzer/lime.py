#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 16 nov. 2017
@author: laurent.vanni@unice.fr
'''

from lime.lime_text import LimeTextExplainer
import random
import numpy as np

class LimeExplainer:

	# ------------------------------
	# INIT
	# ------------------------------
	def __init__(self, model, preprocessing, class_names):
		self.model = model
		self.class_names = class_names
		self.preprocessing = preprocessing
		self.nb_channels = len(preprocessing.channel_texts.keys())
		self.split_expression = " "
		self.explainer = LimeTextExplainer(class_names=class_names, split_expression=self.split_expression)

	# ------------------------------
	# CLASSIFIER
	# ------------------------------
	def classifier_fn(self, text):
		
		X = []

		# MULTI-CHANNELs
		if self.nb_channels > 1:
			for channel in range(self.nb_channels):
				X += [[]]
			for t in text:
				t = t.split(" ")
				for channel in range(self.nb_channels):
					entry = []
					for i, word in enumerate(t):
						if word != "": # LIME word removing algo
							word = word.split("**")[channel]
							entry += [self.preprocessing.dictionaries[channel]["word_index"].get(word, 0)]

					for i in range(len(entry), len(t)):
						entry += [0]
					X[channel] += [entry]
			for channel in range(self.nb_channels):
				X[channel] = np.asarray(X[channel])

		# MONO CHANNEL
		else:
			for t in text:
				entry = []
				for i, word in enumerate(t.split(" ")):
					entry += [self.preprocessing.dictionaries[0]["word_index"].get(word, 0)]
				X += [entry]
			X = np.asarray(X)

		#self.print_data(X)

		P = self.model.predict(X)
		return P


	# ------------------------------
	# CLASSIFIER
	# ------------------------------
	def analyze(self):

		results = []

		# Create an explainer for each text
		for t, text in enumerate(self.preprocessing.raw_texts):
			# log
			print("sample", t+1 , "/" , len(self.preprocessing.raw_texts))
			results += [self.explainer.explain_instance(text, self.classifier_fn, num_features=len(text.split(self.split_expression)), top_labels=len(self.class_names))]

		return results