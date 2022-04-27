#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 27 apr. 2022
@author: laurent.vanni@unice.fr
'''
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, Dense

def computeTDS(config, preprocessing, classifier, x_data, predictions, weighted=False):

	# results
	explainers = []

	# get dictionnaries
	dictionaries = preprocessing.dictionaries

	# -------------------------------------------
	# Inspect model layers:
	# Get convolutionnal layers and dense weights
	i = 0
	conv_layers = []
	attention_layer = []
	dense_weights = []
	dense_bias = []
	for layer in classifier.layers:	
		# CONVOLUTION layers => to compute TDS
		if type(layer) is Conv1D:
			conv_layers += [i+1]

		# DENSE WEIGHTS layers => to compute weighted TDS
		elif type(layer) is Dense:
			dense_weights += [layer.get_weights()[0]]
			dense_bias += [layer.get_weights()[1]]
		i += 1

	# Split the model to get only convultional outputs
	layer_outputs = [layer.output for layer in classifier.layers[len(x_data):conv_layers[-1]]] 
	conv_model = Model(inputs=classifier.input, outputs=layer_outputs)
	conv_outputs = conv_model.predict(x_data)#[-1])

	# Create an explainer for each prediction
	for p, prediction in enumerate(predictions):

		# Tds array that contain the tds scores for each word
		explainer = []

		# log
		if p%100 == 0:
			print("sample", p+1 , "/" , len(predictions))
	
		# Loop on each word
		for w in range(config["SEQUENCE_SIZE"]):
			
			# GET TDS VALUES
			word = []
			for c in range(config["nb_channels"]):

				# -----------------------------------
				# TDS CALCULATION
				# -----------------------------------
				if not weighted: # OLD VERSION (TDS)	
					tds = sum(conv_outputs[-(c+1)][p][w])
					wtds = []
					for classe in config["CLASSES"]:
						wtds += [tds] # Fake wTDS => repeated TDS
				else:
					# NEW VERSION (wTDS)			
					# Get conv output related to the channel c, the prediction p, the word w 
					conv_output = conv_outputs[-(config["nb_channels"]-c)][p][w]
					
					# nb filters of the last conv layer (output size)
					nb_filters = np.size(conv_outputs[-(config["nb_channels"]-c)], 2)

					# Get the weight vector from the first hidden layer
					from_i = c*nb_filters*config["SEQUENCE_SIZE"] # select the sequence
					from_i = from_i + (w*nb_filters) # select the word
					to_j = from_i + nb_filters # end of the vector
					weight1 = dense_weights[0][from_i:to_j,:] # get weight vector

					# Apply weight
					vec = np.dot(conv_output, weight1)# + dense_bias[0]

					# Apply relu function
					vec2 = vec * (vec>0) # RELU

					# Get the weight vector from the last hidden layer
					weight2 = dense_weights[1]

					# Apply weight
					wtds = np.dot(vec2, weight2)# + dense_bias[1]
					wtds = wtds.tolist()
					
				# ADD WORD CHANNEL TDS
				word += [wtds]

			# ADD WORD ENTRY
			explainer += [word]	

		# ADD EXPLAINER ENTRY (one for each prediction)
		explainers.append(explainer)

	return explainers