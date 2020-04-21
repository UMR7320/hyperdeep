import matplotlib.pyplot as plt

import random
import numpy as np
import time
import math
import json

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

class PreProcessing:

	def loadData(self, corpus_file, model_file, config, isTrainingData):   
		
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

			if cpt%100 == 0:
				t1 = time.time()
				print(cpt, "/", len(lines))
				t0 = t1

			# LABELS
			if isTrainingData:
				label = line.split("__ ")[0].replace("__", "")
				"""
				if label not in label_dic.keys():
					label_dic[label] = self.num_classes
					self.num_classes += 1
				label_int = label_dic[label]
				"""
				label_int = config["CLASSES"].index(label)
				labels += [label_int]
				line = line.replace("__" + label + "__ ", "")

			# TEXT
			sequence = []
			for token in line.split():
				self.raw_text += [token]
				args = token.split("**")
				if len(sequence) == 0:
					sequence = [""]*len(args)
				for i, arg in enumerate(args):
					try:
						sequence[i] += arg + " "
					except:
						sequence += ["PAD "]

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

		#data = list(zip(labels, texts))
		#random.shuffle(data)
		#labels, texts = zip(*data)

		dictionaries, datas = tokenize(texts, model_file, isTrainingData, config)

		for i, dictionary in enumerate(dictionaries):
			print('Found %s unique tokens in channel ' % len(dictionary["word_index"]), i+1)

		# Size of validation sample
		nb_validation_samples = int(config["VALIDATION_SPLIT"] * datas[0].shape[0])

		# split the data into a training set and a validation set
		indices = np.arange(datas[0].shape[0])		
		if isTrainingData:
			np.random.shuffle(indices)
			labels = np_utils.to_categorical(np.asarray(labels))
			print('Shape of label tensor:', labels.shape)
			labels = labels[indices]
			self.y_train = labels[:-nb_validation_samples]
			self.y_val = labels[-nb_validation_samples:]

		self.x_train = []
		self.x_val = []
		self.x_data = []
		for data in datas:
			data = data[indices]
			self.x_train += [data[:-nb_validation_samples]]
			self.x_val += [data[-nb_validation_samples:]]

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
		for t in text:
			entry = []
			for i, word in enumerate(t.split(" ")):
				entry += [self.dictionaries[0]["word_index"].get(word, 0)]
			X += [entry]
		X = np.asarray(X)
		P = self.model.predict(X)
		print(P)
		return P
	# ------------------------------

def train(corpus_file, model_file, config):

	if "__TEST__" in model_file:
		
		model_file = model_file.replace("__TEST__", "")

		preprocessing = PreProcessing()
		preprocessing.loadData(corpus_file, model_file, config, isTrainingData = True)

		x_data = []
		for channel in range(len(preprocessing.x_train)):
			x_data += [np.concatenate((preprocessing.x_train[channel],preprocessing.x_val[channel]), axis=0)]
		y_data = np.concatenate((preprocessing.y_train, preprocessing.y_val), axis=0)

		config["num_classes"] = preprocessing.num_classes 
		config["vocab_size"] = []
		for dictionary in preprocessing.dictionaries:
			config["vocab_size"] += [len(dictionary["word_index"])]

		import operator
		import time
		import os

		try:
			os.remove(model_file+"_lime.csv")
			os.remove(model_file+"_z.csv")
			os.remove(model_file+"_tds.csv")
		except:
			pass

		nb_sample = 100
		nb_feature = 10
		for i in range(nb_sample):
			t0 = time.time()

			# GET SENTENCE TO TEST
			preprocessing.x_data = []
			preprocessing.x_data += [np.array(x_data[0][i]).reshape(1, config["SEQUENCE_SIZE"])]

			# GET DEFAUT PREDICTION
			config["ENABLE_LIME"] = True
			tds, classifier, lime = predict(corpus_file, model_file, config, preprocessing)

			
			sentence=""
			for data in preprocessing.x_data[0][0]:
				sentence += preprocessing.dictionaries[0]["index_word"][data] + " "
			
			preprocessing.classifier = classifier
			predicted_class = tds[0][1].index(max(tds[0][1]))
			predicted_score = tds[0][1][predicted_class]
			print("PREDICTED CLASSE:", config["CLASSES"][predicted_class], predicted_score)
			
			config["ENABLE_LIME"] = False
			current_processing_data = preprocessing.x_data
			"""
			sentence=""
			for data in preprocessing.x_data[0][0]:
				sentence += preprocessing.dictionaries[0]["index_word"][data] + " "
			print(sentence)
			"""

			# TEST LIME
			print("-"*50)
			print("LIME")
			lime_csv = open(model_file+"_lime.csv", "a+")
			lime_csv.write(str(i)+"\t"+str(predicted_score)+"\t")
			lime_dic = {}
			for e in lime:
				lime_dic[e[0]] = e[1]
			lime_dic = sorted(lime_dic.items(), key=operator.itemgetter(1), reverse=True)

			for word in lime_dic[:nb_feature]:
				word_id = preprocessing.dictionaries[0]["word_index"][word[0]]
				entry = []
				sentence = []
				for e in preprocessing.x_data[0].reshape(config["SEQUENCE_SIZE"]):
					if e == word_id:
						entry += [0]
					else:
						entry += [e]
					sentence += [preprocessing.dictionaries[0]["index_word"][entry[-1]]]
				#print(" ".join(sentence))
				preprocessing.x_data = []
				preprocessing.x_data += [np.array(entry).reshape(1, config["SEQUENCE_SIZE"])]
				results, _, _ = predict(corpus_file, model_file, config, preprocessing)
				current_score = results[0][1][predicted_class]
				lime_csv.write(str(current_score)+"\t")
				
				#new_sentence=""
				#for data in preprocessing.x_data[0][0]:
				#	new_sentence += preprocessing.dictionaries[0]["index_word"][data] + " "
				#print(new_sentence)
				print("REMOVING:", word[0], "SCORE:", current_score)

			lime_csv.write('\n')
			lime_csv.close()

			# TEST Z-SCORE
			print("-"*50)
			print("Z-SCORE")
			spec = json.load(open(model_file + ".spec", "r"))
			
			preprocessing.x_data = current_processing_data
			"""
			sentence=""
			for data in preprocessing.x_data[0][0]:
				sentence += preprocessing.dictionaries[0]["index_word"][data] + " "
			print(sentence)
			"""

			spec = spec[config["CLASSES"][predicted_class]]["FORME"]
			z_list = {}
			z_csv = open(model_file+"_z.csv", "a+")
			z_csv.write(str(i)+"\t"+str(predicted_score)+"\t")
			for word in sentence:
				try:
					z_list[word] = spec[word]["z"]
				except:
					z_list[word] = 0
			z_list = sorted(z_list.items(), key=operator.itemgetter(1), reverse=True)
			for word in z_list[:nb_feature]:
				word_id = preprocessing.dictionaries[0]["word_index"][word[0]]
				entry = []
				sentence = []
				for e in preprocessing.x_data[0].reshape(config["SEQUENCE_SIZE"]):
					if e == word_id:
						entry += [0]
					else:
						entry += [e]
					sentence += [preprocessing.dictionaries[0]["index_word"][entry[-1]]]
				#print(" ".join(sentence))
				preprocessing.x_data = []
				preprocessing.x_data += [np.array(entry).reshape(1, config["SEQUENCE_SIZE"])]
				results, _, _ = predict(corpus_file, model_file, config, preprocessing)
				current_score = results[0][1][predicted_class]
				z_csv.write(str(current_score)+"\t")

				#new_sentence=""
				#for data in preprocessing.x_data[0][0]:
				#	new_sentence += preprocessing.dictionaries[0]["index_word"][data] + " "
				#print(new_sentence)
				print("REMOVING:", word[0], "SCORE:", current_score)

			z_csv.write('\n')
			z_csv.close()

			# TEST TDS
			print("-"*50)
			print("TDS")
			tds_csv = open(model_file+"_tds.csv", "a+")
			
			preprocessing.x_data = current_processing_data
			sentence=""
			for data in preprocessing.x_data[0][0]:
				sentence += preprocessing.dictionaries[0]["index_word"][data] + " "
			print(sentence)

			tds_csv.write(str(i)+"\t"+str(predicted_score)+"\t")
			tds_list = {}
			for j, word in enumerate(tds[0][0]):
				try:
					word_prev = tds[0][0][j-1]
					prev_tds = word_prev[0][next(iter(word_prev[0]))][predicted_class]
				except:
					prev_tds = 0
				try:
					word_next = tds[0][0][j+1]
					next_tds = word_next[0][next(iter(word_next[0]))][predicted_class]
				except:
					next_tds = 0
				current_tds = word[0][next(iter(word[0]))][predicted_class]

				if (current_tds > prev_tds and current_tds > next_tds): # TDS Peak
					tds_list[j] = current_tds

			tds_list = sorted(tds_list.items(), key=operator.itemgetter(1), reverse=True)

			#for f in range(int(nb_feature/(int(config["FILTER_SIZES"][0])*2-1))):
			for f in range(nb_feature):
				try:
					position = tds_list[f]
				except:
					break
				entry = []
				sentence = []
				word_str = ""
				for j, e in enumerate(preprocessing.x_data[0].reshape(config["SEQUENCE_SIZE"])):
					current_tds = tds[0][0][j][0][next(iter(tds[0][0][j][0]))][predicted_class]
					#if j >= position[0]-2 and j <= position[0]+2: #and current_tds > tds_list[f+1][1]:
					if j == position[0]:
						entry += [0]
						word_str += preprocessing.dictionaries[0]["index_word"][e] + " "
					else:
						entry += [e]
				preprocessing.x_data = []
				preprocessing.x_data += [np.array(entry).reshape(1, config["SEQUENCE_SIZE"])]
				results, _, _ = predict(corpus_file, model_file, config, preprocessing)
				current_score = results[0][1][predicted_class]
				tds_csv.write(str(current_score)+"\t")

				#new_sentence=""
				#for data in preprocessing.x_data[0][0]:
				#	new_sentence += preprocessing.dictionaries[0]["index_word"][data] + " "
				#print(new_sentence)
				print("REMOVING:", word_str, "SCORE:", current_score)	

			tds_csv.write('\n')
			tds_csv.close()

			print(i, "/", nb_sample, time.time() - t0)

		return [0,0]
	
	# preprocess data
	preprocessing = PreProcessing()
	preprocessing.loadData(corpus_file, model_file, config, isTrainingData = True)
	
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
	x_train, y_train, x_val, y_val = preprocessing.x_train, preprocessing.y_train, preprocessing.x_val, preprocessing.y_val
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
				if len(preprocessing.x_train) == len(embedding):
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
	model = load_model(model_file)
	scores = model.evaluate(x_val, y_val, verbose=0)
	return scores
	
def predict(text_file, model_file, config, preprocessing=False):

	result = []

	# GIF ANIMATION
	deconv_images = []

	# preprocess data 
	if not preprocessing: # <== ADDED FOR TESTING
		preprocessing = PreProcessing()
		preprocessing.loadData(text_file, model_file, config, isTrainingData = False)
		x_data = []
		for channel in range(len(preprocessing.x_train)):
			x_data += [np.concatenate((preprocessing.x_train[channel],preprocessing.x_val[channel]), axis=0)]
		classifier = load_model(model_file)
	else:
		x_data = preprocessing.x_data
	try:
		classifier = preprocessing.classifier
	except:
		classifier = load_model(model_file)

	# get dictionnaries
	dictionaries = preprocessing.dictionaries

	#print("----------------------------")
	#print("PREDICTION")
	#print("----------------------------")


	# Plot training & validation accuracy values ---- 
	plot_model(classifier,show_shapes=False, to_file='model.dot')
	plot_model(classifier, to_file='model.png')
	# -----------------------------------------------

	# LIME
	lime = []
	if config["ENABLE_LIME"]:
		predictions = classifier.predict(x_data)
		#print(predictions)
		preprocessing.set_model(classifier)
		explainer = LimeTextExplainer(split_expression=" ")
		for i, data in enumerate(x_data[0]): # Channel 0
			lime_text = ""
			for idx in data:
				lime_text += dictionaries[0]["index_word"][idx] + " "
			lime_text = lime_text[:-1]
			exp = explainer.explain_instance(lime_text, preprocessing.classifier_fn, num_features=config["SEQUENCE_SIZE"], top_labels=config["num_classes"])
			predicted_label = list(predictions[i]).index(max(predictions[i]))
			#print(predictions[i], predicted_label)
			lime += [dict(exp.as_list(label=predicted_label))]
			
			# PRINT RESULTS
			lime_html = open("lime.html", "w")
			lime_html.write(exp.as_html())
			#print(exp.available_labels())
			#print ('\n'.join(map(str, exp.as_list(label=predicted_label))))
		lime = exp.as_list(label=predicted_label)

	#print("----------------------------")
	#print("DECONVOLUTION")
	#print("----------------------------")	
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

	# FLATTEN LAYER
	#layer_outputs = [layer.output for layer in classifier.layers[len(x_data):-3]] 
	#last_model = models.Model(inputs=classifier.input, outputs=layer_outputs)
	#last_model.summary()
	#flatten = last_model.predict(x_data)[-1]

	"""
	# LAST LAYER
	layer_outputs = [layer.output for layer in classifier.layers[len(x_data):-2]] 
	last_model = models.Model(inputs=classifier.input, outputs=layer_outputs)
	last_model.summary()
	last_0 = last_model.predict(x_data)[-1]

	# LAST LAYER
	layer_outputs = [layer.output for layer in classifier.layers[len(x_data):-1]]
	last_model = models.Model(inputs=classifier.input, outputs=layer_outputs)
	last_model.summary()
	last = last_model.predict(x_data)[-1]
	"""

	# TDS LAYERS
	if config["ENABLE_CONV"]:
		layer_outputs = [layer.output for layer in classifier.layers[len(x_data):conv_layers[-1]]] 
		deconv_model = models.Model(inputs=classifier.input, outputs=layer_outputs)
		#print("DECONVOLUTION summary:")
		#deconv_model.summary()
		tds = deconv_model.predict(x_data)#[-1]
	else:
		tds = False

	# ATTENTION LAYER
	"""
	if config["ENABLE_LSTM"]:
		layer_outputs = [layer.output for layer in classifier.layers[len(x_data):attention_layer]] 
		attention_model = models.Model(inputs=classifier.input, outputs=layer_outputs)
		print("ATTENTION summary:")
		attention_model.summary()
		attention = attention_model.predict(x_data)[-1]
	else:
		attention = False
	"""

	# DENSE LAYER 1
	#layer_outputs = [layer.output for layer in classifier.layers[len(x_data):-2]]
	#dense_model = models.Model(inputs=classifier.input, outputs=layer_outputs)
	#dense_model.summary()
	#dense1 = dense_model.predict(x_data)[-1]

	# DENSE LAYER 1
	#print("----------------------------")
	#print("SOFTMAX BREAKDOWN")
	#print("----------------------------")	
	layer_outputs = [layer.output for layer in classifier.layers[len(x_data):-1]]
	dense_model = models.Model(inputs=classifier.input, outputs=layer_outputs)
	#dense_model.summary()
	dense2 = dense_model.predict(x_data)[-1]

	# READ PREDICTION SENTENCE BY SENTENCE
	word_nb = 0
	for sentence_nb in range(len(x_data[0])):

		# CSV
		csv = "Source;Target;Weight;Type\n"
		csv2= "ID;Type\n"

		#print(sentence_nb , "/" , len(x_data[0]))
		sentence = []
		sentence += [[]]
		sentence += [dense2[sentence_nb].tolist()]
		prediction_index = sentence[1].index(max(sentence[1]))

		total = {}

		# READ SENTENCE WORD BY WORD
		pca_array = []
		
		
		for i in range(len(x_data[0][sentence_nb])):
			#print(i , "/" , len(tds[-1][sentence_nb]))

			try:
				attention_value = attention[sentence_nb][i]
			except:
				attention_value = 0
			
			# GET TDS VALUES
			word = []
			for channel in range(preprocessing.nb_channels):
				#print(channel , "/" , len(tds))

				if not tds:
					tds_value = 0
				else:
					# DECONV BY READING FILTERS)

					# OLD TDS
					#tds_value = sum(tds[-(channel+1)][sentence_nb][i])

					"""
					activations = [0]*config["DENSE_LAYER_SIZE"]
					tds_value = 0
					for _i in range(from_i, to_j):
						for _j, weight in enumerate(dense_weights[0][_i]):
							x = int(_i/config["EMBEDDING_DIM"])
							y = _i%config["EMBEDDING_DIM"]
							activations[_j] += tds[-(channel+1)][sentence_nb][x][y]*weight
							#print(_i,_j,x,y,activations[_j])
					
					for _i, activation in enumerate(activations):
						if activation > 0:
							tds_value += activation*dense_weights[1][_i][prediction_index]
					"""

					# NEW TDS
					tds_size = np.size(tds[-1],2) # => nb filters of the last conv layer (output size) (old version : config["EMBEDDING_DIM"])
					tds1 = tds[-(preprocessing.nb_channels-channel)][sentence_nb][i] # <== TEST -1 ???
					from_i = (i*tds_size*preprocessing.nb_channels) + (channel*tds_size)
					to_j = from_i + tds_size
					weight1 = dense_weights[0][from_i:to_j,:]
					vec = np.dot(tds1, weight1) + dense_bias[0]

					"""
					tds_size = np.size(tds[-1],2) # => nb filters of the last conv layer (output size) (old version : config["EMBEDDING_DIM"])
					tds1 = tds[-(preprocessing.nb_channels-channel)][sentence_nb][i]
					from_i = channel*config["EMBEDDING_DIM"]
					vec = np.zeros(len(dense_bias[0]))
					while from_i < (len(x_data[0][sentence_nb])*config["EMBEDDING_DIM"]*preprocessing.nb_channels) - config["EMBEDDING_DIM"]:
						to_j = from_i + config["EMBEDDING_DIM"]
						weight1 = dense_weights[0][from_i:to_j,:]
						vec += np.dot(tds1, weight1) + dense_bias[0]
						from_i += (config["EMBEDDING_DIM"]*preprocessing.nb_channels) + (channel*config["EMBEDDING_DIM"])
					"""

					vec2 = vec * (vec>0) # RELU

					weight2 = dense_weights[1]
					#tds_value = np.dot(vec2, weight2)[prediction_index] + dense_bias[1][prediction_index]
					tds_value = np.dot(vec2, weight2) + dense_bias[1]
					tds_value *= 100


				#print(x_data[channel][sentence_nb].shape)
				index = x_data[channel][sentence_nb][i]

				word_str = dictionaries[channel]["index_word"][index]
				if word_str == "UK":
					word_str = preprocessing.raw_text[word_nb].split("**")[channel]
				word_tds = tds_value.tolist()
				word += [{word_str : word_tds}]

				#word[channel_name]["attention"] = str(attention_value)
				"""
				if config["ENABLE_LIME"]:
					try:
						word[channel_name]["lime"] = lime[sentence_nb][word[channel_name]["str"]]
					except:
						word[channel_name]["lime"] = lime[sentence_nb]["UK"]
				"""
				#print(word[channel_name]["str"], channel, sentence_nb, i, from_i, to_j)
				
				#pca = {}
				#pca[dictionaries[channel]["index_word"][index]] = tds[-(channel+1)][sentence_nb][i].tolist()
				#pca_array += [pca]

				#print(np.transpose(dense_weights[0]).shape)
				#np.transpose(dense_weights[1])

				#weights0 = [dense_weights[0][i][x:x + config["EMBEDDING_DIM"]] for x in range(0, len(dense_weights[0][i]), config["EMBEDDING_DIM"])]
				#print(dense_weights[0][i])

				"""
				for _i, hidden_layer in enumerate(np.transpose(dense_weights[0])):
					
					from_i = config["EMBEDDING_DIM"]*i
					to_j = from_i+config["EMBEDDING_DIM"]
					
					#if word[channel_name]["str"] == "advienne":
					#	print(word[channel_name]["str"], i, " embedding from", from_i, "to", to_j)

					weighted_tds = 0
					sum_tds = 0
					for j, w1 in enumerate(hidden_layer[from_i:to_j]):
						#if word[channel_name]["str"] == "profondément":
						#	print(len(tds[channel][sentence_nb][i]))
						weighted_tds += tds[channel][sentence_nb][i][j]*w1
					if weighted_tds <= 0:
						weighted_tds = 0

					n0 = str(i) + "_" + word[channel_name]["str"]
					n1 = str(_i)
					csv += n0 + ";" + n1 + ";" + str(weighted_tds) + ";Directed\n"
					
					#if word[channel_name]["str"] == "profondément":
					#	print(from_i, to_j, weighted_tds)

					#print("BEFORE:", weighted_tds)
					for _j, w2 in enumerate(dense_weights[1][_i]): 
						weighted_tds = weighted_tds*w2
						#print("AFTER:", weighted_tds)
						csv += n1 + ";" + config["CLASSES"][_j] + ";" + str(weighted_tds) + ";Directed\n"
						
						total[config["CLASSES"][_j]] = total.get(config["CLASSES"][_j], {})
						total[config["CLASSES"][_j]][word[channel_name]["str"]] = total[config["CLASSES"][_j]].get(word[channel_name]["str"], 0) + weighted_tds
						total[config["CLASSES"][_j]]["TOTAL"] = total[config["CLASSES"][_j]].get("TOTAL", 0) + weighted_tds

				csv2 += str(i) + "_" + word[channel_name]["str"] + ";" + "input\n"
				"""

			sentence[0] += [word]	
			word_nb += 1
		
		"""
		for _i in range(len(np.transpose(dense_weights[0]))):
			csv2 += "n" + str(_i) + ";" + "hidden\n"
		for _j in range(len(dense_weights[1][0])):
			csv2 += config["CLASSES"][_j] + ";" + "output\n"
		csv_out = open("links.csv", "w")
		csv_out.write(csv)
		csv_out = open("nodes.csv", "w")
		csv_out.write(csv2)

		# LAYER[-2] ACTIVATION CALCULATION
		total = {}
		print("tds:", tds[-1].shape)
		for _i, hidden_links in enumerate(dense_weights[0]):
			for _j, weight in enumerate(dense_weights[0][_i]):
				x = int(_i/config["EMBEDDING_DIM"])
				y = _i%config["EMBEDDING_DIM"]
				channel = -1
				sentence_nb = 0
				activation = tds[-1][0][x][y]*weight
				total[_j] = total.get(_j, 0) + activation

		activations = []
		for classe, value in total.items():
			if value > 0:
				activations += [value]
			else:
				activations += [0.0]

		print("-"*20)
		print("ACTIVATIONS THEORIQUES:", activations)
		print("ACTIVATIONS OBSERVÉES:", dense1[0].tolist())
		print("-"*20)

		# LAYER[-1] LAYER ACTIVATION CALCULATION
		total = {}
		for _i, hidden_links in enumerate(dense_weights[1]):
			for _j, weight in enumerate(dense_weights[1][_i]):
				activation = activations[_i]*weight
				total[config["CLASSES"][_j]] = total.get(config["CLASSES"][_j], 0) + activation

		activations = []
		for classe, value in total.items():
			activations += [value]


		print("-"*20)
		print("ACTIVATIONS THEORIQUES:", activations)
		print("ACTIVATIONS OBSERVÉES:", dense2[0].tolist())
		print("-"*20)
		"""

		result.append(sentence)

		# ------ DRAW DECONV FACE ------
		"""
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
	return result, classifier, lime 
