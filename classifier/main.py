import random
import numpy as np
import time
import math
import json
import operator
import time
import os

import matplotlib.pyplot as plt

from keras.utils import plot_model
from keras.utils import np_utils
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Conv1D, Conv2D, Conv2DTranspose, TimeDistributed, MaxPooling1D, Dense, Lambda, Flatten
from tensorflow.keras.models import Model

from preprocess.preprocessing import PreProcessing
from classifier.models import Classifier
from analyzer.lime import LimeExplainer

import scipy.misc as smp
import imageio

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
	print("train:", len(y_train), "valid:", y_val, "test:", y_test)

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
	for vocab_size in config["vocab_size"]:
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
	if len(y_test):
		scores = model.evaluate(x_test, y_test, verbose=1)
	else:
		scores = model.evaluate(x_val, y_val, verbose=1)
	print(scores)
	return scores

# ------------------------------
# PREDICT
# ------------------------------
def predict(text_file, model_file, config, preprocessing=False):

	# ------------------------------------------
	# Force to use CPU (no need GPU on predict)
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
	os.environ["CUDA_VISIBLE_DEVICES"] = ""
	# ------------------------------------------

	result = []

	# preprocess data 
	preprocessing = PreProcessing()
	preprocessing.loadData(text_file, model_file, config, getLabels=False, createDictionary=False)
	x_data = []
	for channel in range(len(preprocessing.x_train)):
		x_data += [np.concatenate((preprocessing.x_val[channel],preprocessing.x_test[channel],preprocessing.x_train[channel]), axis=0)]
	classifier = load_model(model_file)

	# get dictionnaries
	dictionaries = preprocessing.dictionaries

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

	print("----------------------------")
	print("TDS")
	print("----------------------------")	
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
		tds = deconv_model.predict(x_data)#[-1]
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

		print(sentence_nb , "/" , len(x_data[0]))

		total = {}	
		
		for i in range(len(x_data[0][sentence_nb])):
			
			# GET TDS VALUES
			word = []
			for channel in range(preprocessing.nb_channels):

				if not tds:
					tds_value = 0
				else:
					# -----------------------------------
					# TDS CALCULATION
					# -----------------------------------
					# OLD VERSION (TDS)
					#tds_value = sum(tds[-(channel+1)][sentence_nb][i])
					#tds_value = [tds_value, tds_value]

					# NEW VERSION (wTDS)
					tds_size = np.size(tds[-1],2) # => nb filters of the last conv layer (output size) (old version : config["EMBEDDING_DIM"])
					tds1 = tds[-(preprocessing.nb_channels-channel)][sentence_nb][i]
					from_i = (i*tds_size*preprocessing.nb_channels) + (channel*tds_size)
					to_j = from_i + tds_size
					weight1 = dense_weights[0][from_i:to_j,:]
					vec = np.dot(tds1, weight1) + dense_bias[0]

					vec2 = vec * (vec>0) # RELU

					weight2 = dense_weights[1]
					#tds_value = np.dot(vec2, weight2)[prediction_index] + dense_bias[1][prediction_index]
					tds_value = np.dot(vec2, weight2) + dense_bias[1]
					tds_value *= 100
					tds_value = tds_value.tolist()
					
				# GET WORD STR
				index = x_data[channel][sentence_nb][i]
				word_str = dictionaries[channel]["index_word"][index]
				if word_str == "__UK__":
					word_str = preprocessing.raw_text[word_nb].split("**")[channel]

				# COMPUTE WORD ENTRY
				word += [{word_str : tds_value}]

			# ADD WORD ENTRY
			sentence[0] += [word]	
			word_nb += 1

		result.append(sentence)

		# ------ DRAW DECONV FACE ------
		"""
		deconv_images = []
		deconv_image = np.zeros( (config["SEQUENCE_SIZE"]*len(x_data), config["EMBEDDING_DIM"], 3), dtype=np.uint8 )
		for channel in range(len(x_data)):
			for y in range(config["SEQUENCE_SIZE"]):
				deconv_value 	= deconv[channel][sentence_nb][y]
				for x in range(int(config["EMBEDDING_DIM"])):
					dv = deconv_value[x]
					dv = dv*200
					deconv_image[y+config["SEQUENCE_SIZE"]*(channel), x] = [dv, dv, dv]

		img = smp.toimage( deconv_image )   # Create a PIL image
		img.save(model_file + ".png")
		deconv_images.append(imageio.imread(model_file + ".png"))

	# CREATE THE GIF ANIMATION
	imageio.mimsave(model_file + ".gif", deconv_images, duration=0.1)
	"""
	return result