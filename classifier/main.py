import random
import numpy as np
import time
import math
import json
import operator
import time
import os
import statistics

import imageio
import scipy.misc as smp
import matplotlib.pyplot as plt

from keras.utils import plot_model
from keras.utils import np_utils
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Conv1D, Conv2D, Conv2DTranspose, TimeDistributed, MaxPooling1D, Dense, Lambda, Flatten
from tensorflow.keras.models import Model

from ..analyzer.lime import LimeExplainer
from ..classifier.preprocessing import PreProcessing
from ..classifier.models import Classifier

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
def train(corpus_file, model_file, config, spec={}):
	
	# preprocess data
	preprocessing = PreProcessing()
	preprocessing.loadData(corpus_file, model_file, config, getLabels=True, createDictionary=True)
	
	if config["SG"] != -1:
		preprocessing.loadEmbeddings(model_file, config)
	else:
		preprocessing.embedding_matrix = None
	
	# Establish params
	config["num_classes"] = preprocessing.num_classes 
	config["nb_channels"] = preprocessing.nb_channels
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
		if not config["TG"][i]: continue;
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
	
	if preprocessing.nb_channels == 1:
		embedding = embedding_model.predict(x_data[0])
	else:
		embedding = embedding_model.predict(x_data)

	# init embeddings
	embeddings = {}
	for channel in range(len(x_data)):
		embeddings[channel] = {}

	# READ ALL SENTENCES (TODO: optimize this!)
	for sentence_nb in range(len(x_data[channel])):
		# READ SENTENCE WORD BY WORD
		for i in range(config["SEQUENCE_SIZE"]):
			# READ EACH CHANNEL
			for channel in range(preprocessing.nb_channels):
				index = x_data[channel][sentence_nb][i]
				word = preprocessing.dictionaries[channel]["index_word"].get(index, "PAD")

				# MUTLI CHANNEL
				if preprocessing.nb_channels > 1:
					wordvector = embedding[channel][sentence_nb][i]

				# ONE CHANNEL
				else:
					wordvector = embedding[sentence_nb][i]
				
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
		tds = computeTDS(config, preprocessing, model, x_test)

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
def computeTDS(config, preprocessing, classifier, x_data):

	print("----------------------------")
	print("TDS")
	print("----------------------------")

	result = []
	# get dictionnaries
	dictionaries = preprocessing.dictionaries

	# GET LAYER INDICES and weights
	i = 0
	conv_layers = []
	attention_layer = []
	dense_weights = []
	dense_bias = []
	for layer in classifier.layers:	
		#print(type(layer), i)
		# CONVOLUTION (AND DECONVOLUTION)
		if type(layer) is Conv1D:
			conv_layers += [i+1]
		# ATTENTION
		elif type(layer) is TimeDistributed:
			attention_layer = i+1
		# DENSE WEIGHTS
		elif type(layer) is Dense:
			dense_weights += [layer.get_weights()[0]]
			dense_bias += [layer.get_weights()[1]]
		i += 1

	# TDS LAYERS
	print("GET TDS...")
	if config["ENABLE_CONV"]:
		layer_outputs = [layer.output for layer in classifier.layers[len(x_data):conv_layers[-1]]] 
		deconv_model = Model(inputs=classifier.input, outputs=layer_outputs)
		print("DECONVOLUTION summary:")
		deconv_model.summary()
		t0 = time.time()
		tds = deconv_model.predict(x_data)#[-1])
	else:
		tds = False

	#----------------------------
	#SOFTMAX BREAKDOWN
	#----------------------------
	layer_outputs = [layer.output for layer in classifier.layers[len(x_data):-1]]
	dense_model = Model(inputs=classifier.input, outputs=layer_outputs)

	t0 = time.time()
	dense2 = dense_model.predict(x_data)[-1]
	#print("dense predict time", time.time() - t0)

	# READ PREDICTION SENTENCE BY SENTENCE
	word_nb = 0
	for sentence_nb in range(len(x_data[0])):
		
		sentence = []
		sentence += [[]]
		sentence += [dense2[sentence_nb].tolist()]
		prediction_index = sentence[1].index(max(sentence[1]))

		if sentence_nb%100 == 0:
			print(sentence_nb , "/" , len(x_data[0]))

		total = {}	
		
		sentence_size = len(x_data[0][sentence_nb])
		for i in range(sentence_size):
			
			# GET TDS VALUES
			word = []
			for channel in range(preprocessing.nb_channels):

				if not tds:
					tds_value = 0
				else:
					# -----------------------------------
					# TDS CALCULATION
					# -----------------------------------
					if config.get("OLD", False):
						# OLD VERSION (TDS)
						_tds_value = sum(tds[-(channel+1)][sentence_nb][i])
						tds_value = []
						for classe in config["CLASSES"]:
							tds_value += [_tds_value]
					else:
						# NEW VERSION (wTDS)
						#print("channel", -(preprocessing.nb_channels-channel))
						tds_size = np.size(tds[-(preprocessing.nb_channels-channel)],2) # => nb filters of the last conv layer (output size) (old version : config["EMBEDDING_DIM"])
						word_tds = tds[-(preprocessing.nb_channels-channel)][sentence_nb][i]
						
						# FIRST HIDDEN LAYER
						vec = False
						from_i = channel*tds_size*sentence_size # OFFSET => Select channel
						to_j = from_i + tds_size*sentence_size
						for t in range(from_i, to_j, tds_size):
							#print(channel, i, t, from_i, to_j)
							weight = dense_weights[0][t:t+tds_size,:]
							try:
								vec += np.dot(word_tds, weight)
							except:
								vec = np.dot(word_tds, weight)
						#from_i = channel*tds_size*sentence_size # OFFSET => Select channel
						#from_i = from_i + (i*tds_size)
						#to_j = from_i + tds_size
						#weight = dense_weights[0][from_i:to_j,:]
						#vec = np.dot(word_tds, weight)
						vec = vec * (vec>0) # RELU

						# LAST HIDDEN LAYER
						tds_value = np.dot(vec, dense_weights[1])						
						tds_value *= 100
						tds_value = tds_value.tolist()

						"""
						tds1 = tds[-(preprocessing.nb_channels-channel)][sentence_nb][i]
						#print(i, tds_size, channel, "/", preprocessing.nb_channels)
						from_i = channel*tds_size*sentence_size # OFFSET => Select channel
						from_i = from_i + (i*tds_size)
						to_j = from_i + tds_size
						#print("from:", from_i)
						#print("to:", to_j)
						weight1 = dense_weights[0][from_i:to_j,:]
						#print(np.shape(tds1), np.shape(weight1), np.shape(dense_bias[0]))
						#print("tds1", tds1)
						#print("weight1", weight1)
						#print("dense_bias", dense_bias)
						vec = np.dot(tds1, weight1)# + dense_bias[0]
						#print("vec", vec)
						#print(np.shape(vec))

						#vec2 = vec * (vec>0) # RELU
						#print("vec2", vec2)
						#print("-"*50)

						#weight2 = dense_weights[1]
						#tds_value = np.dot(vec2, weight2)[prediction_index] + dense_bias[1][prediction_index]
						#tds_value = np.dot(vec2, weight2)# + dense_bias[1]
						#tds_value *= 100
						#tds_value = tds_value.tolist()
						"""
					
				# GET WORD STR
				index = x_data[channel][sentence_nb][i]
				word_str = dictionaries[channel]["index_word"][index]
				if word_str == "__UK__":
					try:
						word_str = preprocessing.raw_text[word_nb].split("**")[channel]
					except:
						print("ERR:", preprocessing.raw_text[word_nb], " channel", channel, "not found")
						word_str = "__UK__"

				# COMPUTE WORD ENTRY
				word += [{word_str : tds_value}]

			# ADD WORD ENTRY
			sentence[0] += [word]	
			word_nb += 1
		result.append(sentence)

	return result

def predict(text_file, model_file, config, preprocessing=False):

	# ------------------------------------------
	# Force to use CPU (no need GPU on predict)
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
	os.environ["CUDA_VISIBLE_DEVICES"] = ""
	# ------------------------------------------

	# preprocess data 
	preprocessing = PreProcessing()
	preprocessing.loadData(text_file, model_file, config, getLabels=False, createDictionary=False)
	x_data = []
	for channel in range(len(preprocessing.x_train)):
		x_data += [np.concatenate((preprocessing.x_val[channel],preprocessing.x_test[channel],preprocessing.x_train[channel]), axis=0)]
	classifier = load_model(model_file)

	print("----------------------------")
	print("PREDICTION")
	print("----------------------------")

	# Plot training & validation accuracy values ---- 
	#plot_model(classifier,show_shapes=False, to_file='model.dot')
	#plot_model(classifier, to_file='model.png')
	# -----------------------------------------------

	# LIME
	if config["ENABLE_LIME"]:
		print("----------------------------")
		print("LIME")
		print("----------------------------")
		limeExplainer = LimeExplainer(preprocessing, classifier)
		lime = limeExplainer.analyze(x_data)

	# GET TDS scores from predictions
	predictions = computeTDS(config, preprocessing, classifier, x_data)

	return predictions