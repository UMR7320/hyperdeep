#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 16 nov. 2017
@author: laurent.vanni@unice.fr
'''

from lime.lime_text import LimeTextExplainer
import random

class LimeExplainer:

	# ------------------------------
	# INIT
	# ------------------------------
	def __init__(self, preprocessing, model):
		self.preprocessing = preprocessing
		self.model = model

	# ------------------------------
	# TOOLS
	# ------------------------------
	def print_data(self, data, preprocessing):
		for i, sentence in enumerate(data[0]):
			sentence_to_String = ""
			for j, w in enumerate(sentence):
				for channel in range(3):
					word = data[channel][i][j]
					sentence_to_String += preprocessing.dictionaries[channel]["index_word"][word] + "**"
				sentence_to_String = sentence_to_String.strip("**") + " "

			print(sentence_to_String)
			print("-"*50)

	# ------------------------------
	# CLASSIFIER
	# ------------------------------
	def classifier_fn(self, text):
		
		X = []

		# MULTI-CHANNELs
		if self.nb_channels > 1:
			for channel in range(self.preprocessing.nb_channels):
				X += [[]]
			for t in text:
				t = t.split(" ")
				for channel in range(self.preprocessing.nb_channels):
					entry = []
					for i, word in enumerate(t):
						if word != "": # LIME word removing algo
							word = word.split("**")[channel]
							entry += [self.preprocessing.dictionaries[channel]["word_index"].get(word, 0)]

					for i in range(len(entry), len(t)):
						entry += [0]
					X[channel] += [entry]
			for channel in range(self.preprocessing.nb_channels):
				X[channel] = np.asarray(X[channel])

		# MONO CHANNEL
		else:
			for t in text:
				entry = []
				for i, word in enumerate(t.split(" ")):
					entry += [self.preprocessing.dictionaries[0]["word_index"].get(word, 0)]
				X += [entry]
			X = np.asarray(X)

		#print_data(X, self)

		P = self.model.predict(X)
		return P


	# ------------------------------
	# CLASSIFIER
	# ------------------------------
	def analyse(self, x_data):
		lime = []
		predictions = self.model.predict(x_data)
		explainer = LimeTextExplainer(split_expression=" ")
		for sentence_nb in range(len(x_data[0])): # Channel 0
			lime_text = ""
			for i in range(len(x_data[0][sentence_nb])):
				for channel in range(preprocessing.nb_channels):
					idx = x_data[channel][sentence_nb][i]
					lime_text += dictionaries[channel]["index_word"][idx] + "**"
				lime_text = lime_text.strip("**") + " "
			lime_text = lime_text[:-1]
			exp = explainer.explain_instance(lime_text, limeExplainer.classifier_fn, num_features=config["SEQUENCE_SIZE"], top_labels=config["num_classes"])
			predicted_label = list(predictions[sentence_nb]).index(max(predictions[sentence_nb]))
			#print(predictions[i], predicted_label)
			lime += [dict(exp.as_list(label=predicted_label))]
			
			# PRINT RESULTS
			lime_html = open("lime.html", "w")
			lime_html.write(exp.as_html())
			print(exp.available_labels())
			print ('\n'.join(map(str, exp.as_list(label=predicted_label))))

		# RETURN RESULTS	
		lime_list = exp.as_list(label=predicted_label)
		lime = {}
		for e in lime_list:
			lime[e[0]] = e[1]
		return lime