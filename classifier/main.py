import random
import numpy as np
import time
import math
import json
import operator
import time
import os
import statistics
import platform

import imageio
import scipy.misc as smp
import matplotlib.pyplot as plt

from keras.utils import plot_model
from keras.utils import np_utils
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Conv1D, Conv2D, Conv2DTranspose, TimeDistributed, MaxPooling1D, Dense, Lambda, Flatten
from tensorflow.keras.models import Model

from analyzer.lime import LimeExplainer
from classifier.preprocessing import PreProcessing
from classifier.models import Classifier

# ------------------------------
# Visualization tools
# ------------------------------
def plot_history(history):
	plt.plot(history.history['loss'])
	plt.plot(history.history['accuracy'])
	plt.plot(history.history['val_loss'])
	plt.plot(history.history['val_accuracy'])
	plt.title('Model loss and accuracy')
	plt.ylabel('Loss/Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['train_loss', 'train_acc', 'val_loss', 'val_acc'], loc='upper right')
	#plt.show()
	plt.savefig(model_file + ".png")

# ------------------------------
# TRAIN
# ------------------------------
def train(corpus_file, model_file, config):
	
	# preprocess data
	preprocessing = PreProcessing()
	preprocessing.loadData(corpus_file, model_file, config)
	
	if config["SG"] != -1:
		preprocessing.loadEmbeddings(model_file, config)
	else:
		preprocessing.embedding_matrix = None
	
	# Establish params
	config["num_classes"] = preprocessing.num_classes 
	config["vocab_size"] = []
	for dictionary in preprocessing.dictionaries:
		config["vocab_size"] += [len(dictionary["word_index"])]

	# GET TRAIN DATASET
	x_train, y_train, x_val, y_val, x_test, y_test = preprocessing.x_train, preprocessing.y_train, preprocessing.x_val, preprocessing.y_val, preprocessing.x_test, preprocessing.y_test
	print("Available samples:")
	print("train:", len(x_train[0]), "valid:", len(x_val[0]), "test:", len(x_test[0]))

	checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	earlystop = EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=3, verbose=1, mode='max')
	callbacks_list = [checkpoint, earlystop]

	# create and get model
	classifier = Classifier()
	model = classifier.getModel(config=config, weight=preprocessing.embedding_matrix)
	history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=config["NUM_EPOCHS"], batch_size=config["BACH_SIZE"], callbacks=callbacks_list)

	# Plot training & validation loss values
	# plot_history(history)

	# ------------------------------------
	# GET EMBEDDING MODEL
	print("-"*50)
	print("EMBEDDING CALCULATION...")
	layer_outputs = [layer.output for layer in model.layers[len(x_train):len(x_train)*2]] 
	embedding_model = Model(inputs=model.input, outputs=layer_outputs)
	embedding_model.summary()

	# GET WORD EMBEDDINGS
	x_data = []
	for i, vocab_size in enumerate(config["vocab_size"]):

		x_entry = []
		entry = []
		for word_index in range(config["vocab_size"][0]):
			if word_index%config["SEQUENCE_SIZE"] == 0 and word_index != 0:
				x_entry.append(entry)
				entry = []
			entry += [word_index%vocab_size]

		for word_index in range(config["SEQUENCE_SIZE"]-len(entry)):
			entry += [0]
		x_entry.append(entry)
		x_data += [np.array(x_entry)]
	
	if config["nb_channels"] == 1:
		embedding = embedding_model.predict(x_data[0])
	else:
		embedding = embedding_model.predict(x_data)

	# init embeddings
	embeddings = {}
	for channel in range(len(x_data)):
		embeddings[channel] = {}

	# READ ALL SENTENCES (TODO: optimize this!)
	for p in range(len(x_data[channel])):
		# READ SENTENCE WORD BY WORD
		for i in range(config["SEQUENCE_SIZE"]):
			# READ EACH CHANNEL
			for channel in range(config["nb_channels"]):
				index = x_data[channel][p][i]
				word = preprocessing.dictionaries[channel]["index_word"].get(index, "PAD")

				# MUTLI CHANNEL
				if config["nb_channels"] > 1:
					wordvector = embedding[channel][p][i]

				# ONE CHANNEL
				else:
					wordvector = embedding[p][i]
				
				embeddings[channel][word] = wordvector

	for channel in range(len(x_data)):
		f = open(model_file + ".finalvec" + str(channel) ,'w')
		vectors = []
		vector = '{} {}\n'.format(len(embeddings[channel].keys()), config["EMBEDDING_DIM"])
		vectors.append(vector)
		f.write(vector)    
		for word, values in embeddings[channel].items():
			vector = word + " " + " ".join([str(f) for f in values]) + "\n"
			vectors.append(vector)
			f.write(vector)
		f.flush()
		f.close()
	print("DONE.")
	
	# ------------------------------------
	# get score
	print("-"*50)
	print("TESTING")
	print(len(y_train), len(y_val), len(y_test))
	model = load_model(model_file)
	
	# SMALL TEST
	if not len(y_test):
		scores = model.evaluate(x_val, y_val, verbose=1)

	# LARGE TEST
	else: 
		scores = model.evaluate(x_test, y_test, verbose=1)

		# COMPUTE TDS ON TEST DATASET
		predictions = dense_model.predict(x_test)[-1]
		tds = computeTDS(config, preprocessing, model, x_test, predictions, weighted=method=="wTDS")

		results = {}
		accurracy = {}
		nb_words = {}
		classes = config["CLASSES"]
		for entry in tds:
	        
			# PREDICTED CLASS
			classe_value = max(entry[1])
			classe_id = entry[1].index(classe_value) # predicted_class
			classe_name = classes[classe_id]
			results[classe_name] = results.get(classe_name, {})

			accurracy[classe_name] = accurracy.get(classe_name, {})
			accurracy[classe_name]["score"] = accurracy[classe_name].get("score", 0) + classe_value
			accurracy[classe_name]["taille"] = accurracy[classe_name].get("taille", 0) + 1

			for i, channel in enumerate(range(len(entry[0][0]))):
				results[classe_name][i] = results[classe_name].get(i, 0)
				for word in entry[0]:
					word_str = next(iter(word[channel]))
					word_tds = word[channel][word_str][classe_id]

					#if word_tds > results[classe_name][i][-1]:
					#	results[classe_name][i][-1] = word_tds
					results[classe_name][i] += word_tds
					nb_words[classe_name] = nb_words.get(classe_name, {})
					nb_words[classe_name][i] = nb_words[classe_name].get(i, 0) + 1

		for classe_name in classes:
			try:
				print(classe_name, accurracy[classe_name]["score"]/accurracy[classe_name]["taille"])
				try:
					for i, value in results[classe_name].items():
						print(results[classe_name][i]/nb_words[classe_name][i], end="\t")
				except:
					print(0, end="\t")
				print("\n" + "-"*5)
			except:
				print(classe_name, "no data...")
				pass
	return scores

# ------------------------------
# PREDICT
# ------------------------------
def computeTDS(config, preprocessing, classifier, x_data, predictions, weighted=False):

	explainers = []
	# get dictionnaries
	dictionaries = preprocessing.dictionaries

	# GET LAYER INDICES and weights
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

def predict(text_file, model_file, config):

	# ------------------------------------------
	# Force to use CPU (no need GPU on predict)
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
	os.environ["CUDA_VISIBLE_DEVICES"] = ""

	# ------------------------------------------
	# Preprocess data 
	preprocessing = PreProcessing()
	preprocessing.loadData(text_file, model_file, config)

	x_data = []
	for channel in range(len(preprocessing.x_train)):
		x_data += [np.concatenate((preprocessing.x_val[channel],preprocessing.x_test[channel],preprocessing.x_train[channel]), axis=0)]
	classifier = load_model(model_file)

	# ------------------------------------------
	print("-"*50)
	print("PREDICTION")
	print("-"*50)
	
	# SOFTMAX BREAKDOWN
	if config["SOFTMAX_BREAKDOWN"]:
		layer_outputs = [layer.output for layer in classifier.layers[len(x_data):-1]]
		classifier = Model(inputs=classifier.input, outputs=layer_outputs)
		predictions = classifier.predict(x_data)[-1]
	else:
		predictions = classifier.predict(x_data)

	# Predictions
	predictions_list = []
	for p, prediction in enumerate(predictions):
		prediction = prediction.tolist()
		classe_value = max(prediction)
		classe_id = prediction.index(classe_value)
		classe_name = config["CLASSES"][classe_id]
		predictions_list += [(classe_name, classe_id)]
		print("sample", p+1, ":", classe_name, round(classe_value, 2))

	# ------------------------------------------
	# EXPLANATION
	explaners = {}
	for method in config["ANALYZER"]:

		# LOG
		print("-"*50)
		print(method)
		print("-"*50)

		# ------------------------------------------
		# TDS
		if method in ["TDS", "wTDS"]:
			tds_explainers = computeTDS(config, preprocessing, classifier, x_data, predictions, weighted=method=="wTDS")
			explaners[method] = tds_explainers

			# -----------------------------------------------------
			# OUTPUT LOG : pretty print on first sample
			fig, ax = plt.subplots()
			index = np.arange(config["SEQUENCE_SIZE"])
			bar_width = 1/config["nb_channels"]
			colors = ['r', 'g', 'b']
			words = []
			for c in range(config["nb_channels"]):
				tds_values = []
				for w, word in enumerate(tds_explainers[0]): # first sample
					if c == 0:
						words += [preprocessing.channel_texts[c][0][w]]
					tds_values += [tds_explainers[0][w][c][predictions_list[0][1]]]
				plt.bar(index + bar_width*c, tds_values, bar_width, color=colors[c], label='Channel' + str(c))

			plt.ylabel(method)
			plt.title(method + " for " + predictions_list[0][0])
			plt.xticks(index + (bar_width*config["nb_channels"])/2, words, rotation=90)
			plt.legend()

			plt.show()

		# ------------------------------------------
		# LIME
		if method == "LIME":
			limeExplainer = LimeExplainer(classifier, preprocessing, config["CLASSES"])
			lime_explainers = limeExplainer.analyze()

			fitted_explainer = []
			for t, text in enumerate(preprocessing.raw_texts):
				text_explainer = []
				explainer_dict = {}

				for classe in range(len(config["CLASSES"])):
					for word, score in lime_explainers[t].as_list(classe):
						explainer_dict[word] = explainer_dict.get(word, []) + [score]
				
				for w, word in enumerate(text.split(limeExplainer.split_expression)):
					word_explainer = []
					for c in range(config["nb_channels"]):
						word_explainer += [explainer_dict[word]]
					text_explainer += [word_explainer]
				fitted_explainer += [text_explainer]
			explaners[method] = fitted_explainer

			# -----------------------------------------------------
			# OUTPUT LOG : pretty print on first sample
			# lime_explainers[0].as_pyplot_figure()
			# plt.tight_layout()
			#plt.show()
			lime_html = lime_explainers[0].as_html()
			open("lime.html", "w").write(lime_html)
			try:
				print("Trying to open lime.html...")
				if platform.system() == "Windows":
					os.system("start lime.html")
				else:
					os.system("open lime.html")
			except:
				print("Failed.")
				print("Open lime.html file manualy to see lime explainer")

	# ------------------------------------------
	# COMPUTE RESULTS
	results = []
	for t in range(len(preprocessing.raw_texts)):
		result = []
		result += [[]]
		result += [predictions.tolist()[t]]
		for c in range(config["nb_channels"]):
			channel = []
			for w in range(config["SEQUENCE_SIZE"]):
				word = {}
				try:
					word_str = preprocessing.channel_texts[c][t][w]
				except:
					word_str = "UK"
				word[word_str] = {}
				for method in config["ANALYZER"]:
					word[word_str][method] = explaners[method][t][w][c]
				channel += [word]
			result[0] += [channel]
		results += [result]

	return results