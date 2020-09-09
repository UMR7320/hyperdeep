import matplotlib.pyplot as plt

import random
import numpy as np
import time
import math
import json
import operator
import time
import os

from keras.utils import plot_model
from keras.utils import np_utils
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv1D, Conv2D, Conv2DTranspose, TimeDistributed, MaxPooling1D, Dense, Lambda, Flatten
from keras.models import Model

from classifier.cnn import models
from skipgram.skipgram_with_NS import create_vectors
from data_helpers import tokenize
import scipy.misc as smp
import imageio

from lime.lime_text import LimeTextExplainer

# FOR TEST
from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
import tensorflow
import gc
from pympler.tracker import SummaryTracker

class PreProcessing:

	def loadData(self, corpus_file, model_file, config, getLabels, createDictionary):   
		
		print("loading data...")
		
		f = open(corpus_file, "r")
		lines = f.readlines()
		self.corpus_file = corpus_file
		self.raw_text = []

		label_dic = {}
		labels = []
		texts = {}
		cpt = 0
		t0 = time.time()

		for line in lines:
			if "--" in line: continue

			if cpt%100 == 0:
				t1 = time.time()
				print(cpt, "/", len(lines))
				t0 = t1

			# LABELS
			if getLabels:
				label = line.split("__ ")[0].replace("__", "")
				label_int = config["CLASSES"].index(label)
				labels += [label_int]
				line = line.replace("__" + label + "__ ", "")

			# TEXT
			if config["TG"]:
				sequence = ["", "", ""] # MULTICHANNELS
			else:
				sequence = [""] # MONOCHANNEL
			for token in line.split():
				self.raw_text += [token]
				args = token.split("**")
				for i in range(len(sequence)):
					try:
						if not args[i]:
							sequence[i] += "PAD "
						else:
							sequence[i] += args[i] + " "
					except:
						sequence[i] += "PAD "

			for i in range(len(sequence)):
				texts[i] = texts.get(i, [])
				texts[i].append(sequence[i])
		
			cpt += 1
		f.close()

		for i, text in texts.items():
			f = open(corpus_file + "." + str(i), "w")
			for sequence in text:
				f.write(sequence + "\n")
			f.close()
		
		#print("DETECTED LABELS :")
		#print(label_dic)
		self.num_classes = len(config["CLASSES"])

		dictionaries, datas = tokenize(texts, model_file, createDictionary, config)

		for i, dictionary in enumerate(dictionaries):
			print('Found %s unique tokens in channel ' % len(dictionary["word_index"]), i+1)

		# Size of each dataset (train, valid, test)
		nb_validation_samples = int(config["VALIDATION_SPLIT"] * datas[0].shape[0])
		nb_testing_samples = nb_validation_samples + int(config["TESTING_SPLIT"] * datas[0].shape[0])

		# split the data into a training set and a validation set
		indices = np.arange(datas[0].shape[0])		
		if getLabels:
			np.random.shuffle(indices)
			labels = np_utils.to_categorical(np.asarray(labels))
			print('Shape of label tensor:', labels.shape)
			labels = labels[indices]
			self.y_val = labels[:nb_validation_samples]
			self.y_test = labels[nb_validation_samples:nb_testing_samples]
			self.y_train = labels[nb_testing_samples:]

		self.x_train = []
		self.x_val = []
		self.x_test = []
		self.x_data = []
		for data in datas:
			data = data[indices]
			self.x_val += [data[:nb_validation_samples]]
			self.x_test += [data[nb_validation_samples:nb_testing_samples]]
			self.x_train += [data[nb_testing_samples:]]

		self.dictionaries = dictionaries
		self.nb_channels = len(texts.keys())

	def loadEmbeddings(self, model_file, config, create_v = False):

		print("LOADING WORD2VEC EMBEDDING")
		
		self.embedding_matrix = []

		if not create_v:
			create_vectors(self.corpus_file, model_file, config, nb_channels=self.nb_channels)

		for i in range(self.nb_channels):
			my_dictionary = self.dictionaries[i]["word_index"]
			embeddings_index = {}
			vectors = open(model_file + ".word2vec" + str(i) ,'r')
				
			for line in vectors.readlines():
				values = line.split()
				word = values[0]
				coefs = np.asarray(values[1:], dtype='float32')
				embeddings_index[word] = coefs

			print('Found %s word vectors.' % len(embeddings_index))
			self.embedding_matrix += [np.zeros((len(my_dictionary), config["EMBEDDING_DIM"]))]
			for word, j in my_dictionary.items():
				embedding_vector = embeddings_index.get(word)
				if embedding_vector is not None:
					# words not found in embedding index will be all-zeros.
					self.embedding_matrix[i][j] = embedding_vector
			vectors.close()

	# ------------------------------
	# LIME
	def set_model(self, model):
		self.model = model

	import random
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
							entry += [self.dictionaries[channel]["word_index"].get(word, 0)]

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
					entry += [self.dictionaries[0]["word_index"].get(word, 0)]
				X += [entry]
			X = np.asarray(X)

		#print_data(X, self)

		P = self.model.predict(X)
		return P

# ------------------------------
# TOOLS
# ------------------------------
def print_data(data, preprocessing):
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
	config["vocab_size"] = []
	for dictionary in preprocessing.dictionaries:
		config["vocab_size"] += [len(dictionary["word_index"])]

	# GET TRAIN DATASET
	x_train, y_train, x_val, y_val, x_test, y_test = preprocessing.x_train, preprocessing.y_train, preprocessing.x_val, preprocessing.y_val, preprocessing.x_test, preprocessing.y_test
	print("Available samples:")
	print("train:", len(y_train), "valid:", y_val, "test:", y_test)

	checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]

	# create and get model
	cnn_model = models.CNNModel()
	model = cnn_model.getModel(config=config, weight=preprocessing.embedding_matrix)
	history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=config["NUM_EPOCHS"], batch_size=config["BACH_SIZE"], callbacks=callbacks_list)

	# Plot training & validation loss values
	plt.plot(history.history['loss'])
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_loss'])
	plt.plot(history.history['val_acc'])
	plt.title('Model loss and accuracy')
	plt.ylabel('Loss/Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['train_loss', 'train_acc', 'val_loss', 'val_acc'], loc='upper right')
	#plt.show()
	plt.savefig(model_file + ".png")

	# ------------------------------------
	# GET EMBEDDING MODEL
	print("-"*50)
	print("EMBEDDING CALCULATION...")
	layer_outputs = [layer.output for layer in model.layers[len(x_train):len(x_train)*2]] 
	embedding_model = models.Model(inputs=model.input, outputs=layer_outputs)
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
	
	if not config["TG"]:
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
			for channel in range(len(x_data)):
				index = x_data[channel][sentence_nb][i]
				word = preprocessing.dictionaries[channel]["index_word"].get(index, "PAD")

				# MUTLI CHANNEL
				if config["TG"]:
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
	lime = []
	if config["ENABLE_LIME"]:
		predictions = classifier.predict(x_data)
		preprocessing.set_model(classifier)
		explainer = LimeTextExplainer(split_expression=" ")
		for sentence_nb in range(len(x_data[0])): # Channel 0
			lime_text = ""
			for i in range(len(x_data[0][sentence_nb])):
				for channel in range(preprocessing.nb_channels):
					idx = x_data[channel][sentence_nb][i]
					lime_text += dictionaries[channel]["index_word"][idx] + "**"
				lime_text = lime_text.strip("**") + " "
			lime_text = lime_text[:-1]
			exp = explainer.explain_instance(lime_text, preprocessing.classifier_fn, num_features=config["SEQUENCE_SIZE"], top_labels=config["num_classes"])
			predicted_label = list(predictions[sentence_nb]).index(max(predictions[sentence_nb]))
			#print(predictions[i], predicted_label)
			lime += [dict(exp.as_list(label=predicted_label))]
			
			# PRINT RESULTS
			lime_html = open("lime.html", "w")
			lime_html.write(exp.as_html())
			print(exp.available_labels())
			print ('\n'.join(map(str, exp.as_list(label=predicted_label))))
		lime_list = exp.as_list(label=predicted_label)
		lime = {}
		for e in lime_list:
			lime[e[0]] = e[1]

	print("----------------------------")
	print("DECONVOLUTION")
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
		deconv_model = models.Model(inputs=classifier.input, outputs=layer_outputs)
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
	dense_model = models.Model(inputs=classifier.input, outputs=layer_outputs)

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

# ------------------------------
# TEST DATASET
# ------------------------------
def test(corpus_file, model_file, config):

	tracker = SummaryTracker()

	model_file = model_file.replace("__TEST__", "")
	config = json.loads(open(model_file + ".config", "r").read())
	classifier = load_model(model_file)

	preprocessing = PreProcessing()
	preprocessing.loadData(corpus_file, model_file, config, getLabels=True, createDictionary=False)
	preprocessing.classifier = classifier

	x_data = []
	for channel in range(len(preprocessing.x_train)):
		x_data += [np.concatenate((preprocessing.x_val[channel], preprocessing.x_test[channel], preprocessing.x_train[channel]), axis=0)]
	y_data = np.concatenate((preprocessing.y_val, preprocessing.y_test, preprocessing.y_train), axis=0)

	scores = classifier.evaluate(x_data, y_data, verbose=1)
	print(scores)
	return scores