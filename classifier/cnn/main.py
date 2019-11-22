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

	def loadData(self, corpus_file, model_file, config, create_dictionnary):   
		
		print("loading data...")
		
		self.corpus_file = corpus_file
		
		label_dic = {}
		labels = []
		texts = {}
		
		# Read text and detect classes/labels
		self.num_classes = 0
		
		f = open(corpus_file, "r")
		lines = f.readlines()
		cpt = 0
		t0 = time.time()
		for line in lines:

			if cpt%100 == 0:
				t1 = time.time()
				print(cpt, "/", len(lines))
				t0 = t1

			# LABELS
			label = line.split("__ ")[0].replace("__", "")
			if label not in label_dic.keys():
				label_dic[label] = self.num_classes
				self.num_classes += 1
			label_int = label_dic[label]
			labels += [label_int]

			# TEXT
			line = line.replace("__" + label + "__ ", "")
			sequence = []
			for token in line.split():
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
		
		print("DETECTED LABELS :")
		#print(label_dic)

		#data = list(zip(labels, texts))
		#random.shuffle(data)
		#labels, texts = zip(*data)

		dictionaries, datas = tokenize(texts, model_file, create_dictionnary, config)
		#print(dictionaries[1])

		for i, dictionary in enumerate(dictionaries):
			print('Found %s unique tokens in channel ' % len(dictionary["word_index"]), i+1)

		labels = np_utils.to_categorical(np.asarray(labels))
		print('Shape of label tensor:', labels.shape)

		# Size of validation sample
		nb_validation_samples = int(config["VALIDATION_SPLIT"] * datas[0].shape[0])

		# SHUFFLE
		indices = np.arange(datas[0].shape[0])
		if create_dictionnary: # ONLY FOR TRAINING
			np.random.shuffle(indices)

		# split the data into a training set and a validation set		
		labels = labels[indices]
		self.y_train = labels[:-nb_validation_samples]
		self.y_val = labels[-nb_validation_samples:]
		self.x_train = []
		self.x_val = []
		for data in datas:
			data = data[indices]
			self.x_train += [data[:-nb_validation_samples]]
			self.x_val += [data[-nb_validation_samples:]]

		self.dictionaries = dictionaries
		self.nb_channels = len(texts.keys())

	def loadEmbeddings(self, model_file, config, create_v = False):
		
		self.embedding_matrix = []

		if not create_v:
			create_vectors(self.corpus_file, model_file, config, nb_channels=self.nb_channels)

		for i in range(self.nb_channels):
			my_dictionary = self.dictionaries[i]["word_index"]
			embeddings_index = {}
			vectors = open(model_file + ".vec" + str(i) ,'r')
				
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

	# preprocess data
	preprocessing = PreProcessing()
	preprocessing.loadData(corpus_file, model_file, config, create_dictionnary = True)
	
	if config["SG"] != -1:
		preprocessing.loadEmbeddings(model_file, config)
	else:
		preprocessing.embedding_matrix = None
	
	# Establish params
	config["num_classes"] = preprocessing.num_classes 
	config["vocab_size"] = []
	for dictionary in preprocessing.dictionaries:
		config["vocab_size"] += [len(dictionary["word_index"])]

	# create and get model
	cnn_model = models.CNNModel()
	model = cnn_model.getModel(config=config, weight=preprocessing.embedding_matrix)

	# train model
	x_train, y_train, x_val, y_val = preprocessing.x_train, preprocessing.y_train, preprocessing.x_val, preprocessing.y_val
	checkpoint = ModelCheckpoint(model_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	callbacks_list = [checkpoint]
	
	model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=config["NUM_EPOCHS"], batch_size=config["BACH_SIZE"], callbacks=callbacks_list)

	"""
	# ------------------------------------
	# GET EMBEDDING MODEL
	layer_outputs = [layer.output for layer in model.layers[len(x_train):len(x_train)*2]] 
	embedding_model = models.Model(inputs=model.input, outputs=layer_outputs)
	embedding_model.summary()
	
	# GET WORD EMBEDDINGS
	x_data = []
	for channel in range(len(preprocessing.x_train)):
		x_data += [np.concatenate((preprocessing.x_train[channel],preprocessing.x_val[channel]), axis=0)]
	embedding = embedding_model.predict(x_data)
	
	# get dictionnaries
	dictionaries = preprocessing.dictionaries
	
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
				word = dictionaries[channel]["index_word"].get(index, "PAD")

				# MUTLI CHANNEL
				if len(preprocessing.x_train) == len(embedding):
					wordvector = embedding[channel][sentence_nb][i]

				# ONE CHANNEL
				else:
					wordvector = embedding[sentence_nb][i]
				
				embeddings[channel][word] = wordvector

	for channel in range(len(x_data)):
		f = open(model_file + ".vec" + str(channel) ,'w')
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
	"""
	# ------------------------------------

	# get score
	model = load_model(model_file)
	scores = model.evaluate(x_val, y_val, verbose=0)
	return scores
	
def predict(text_file, model_file, config, vectors_file):

	result = []

	# GIF ANIMATION
	deconv_images = []

	# preprocess data
	preprocessing = PreProcessing()
	preprocessing.loadData(text_file, model_file, config, create_dictionnary = False)
	raw_text = open(text_file, "r").read().split()

	# get dictionnaries
	dictionaries = preprocessing.dictionaries

	print("----------------------------")
	print("PREDICTION")
	print("----------------------------")
	classifier = load_model(model_file)

	plot_model(classifier,show_shapes=False, to_file='model.dot')
	plot_model(classifier, to_file='model.png')
	
	x_data = []
	for channel in range(len(preprocessing.x_train)):
		x_data += [np.concatenate((preprocessing.x_train[channel],preprocessing.x_val[channel]), axis=0)]
	
	predictions = classifier.predict(x_data)
	print(predictions)

	# LIME
	if config["ENABLE_LIME"]:
		lime = []
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
			#print(exp.available_labels())
			#print ('\n'.join(map(str, exp.as_list(label=4))))

	print("----------------------------")
	print("DECONVOLUTION")
	print("----------------------------")	
	# GET LAYER INDICES and weights
	i = 0
	conv_layers = []
	attention_layer = []
	dense_weights = []
	for layer in classifier.layers:	
		print(type(layer), i)
		# CONVOLUTION (AND DECONVOLUTION)
		if type(layer) is Conv1D:
			conv_layers += [i+1]
		# ATTENTION
		elif type(layer) is TimeDistributed:
			attention_layer = i+1
		# DENSE WEIGHTS
		elif type(layer) is Dense:
			dense_weights += [layer.get_weights()[0]]
		i += 1

	"""
	# FLATTEN LAYER
	layer_outputs = [layer.output for layer in classifier.layers[len(x_data):-3]] 
	last_model = models.Model(inputs=classifier.input, outputs=layer_outputs)
	last_model.summary()
	flatten = last_model.predict(x_data)[-1]

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

		# DECONV BY CONV2DTRANSPOSE
		try:
			tds = []
			for channel in range(preprocessing.nb_channels):
				deconv_model = load_model(model_file + ".deconv" + str(channel))
				print("DECONVOLUTION summary:")
				deconv_model.summary()
				tds += [deconv_model.predict(x_data[channel])]
		
		# DECONV BY READING FILTERS
		except:
			layer_outputs = [layer.output for layer in classifier.layers[len(x_data):conv_layers[-1]]] 
			deconv_model = models.Model(inputs=classifier.input, outputs=layer_outputs)
			print("DECONVOLUTION summary:")
			deconv_model.summary()
			tds = deconv_model.predict(x_data)#[-1]
	else:
		tds = False

	# ATTENTION LAYER
	if config["ENABLE_LSTM"]:
		layer_outputs = [layer.output for layer in classifier.layers[len(x_data):attention_layer]] 
		attention_model = models.Model(inputs=classifier.input, outputs=layer_outputs)
		print("ATTENTION summary:")
		attention_model.summary()
		attention = attention_model.predict(x_data)[-1]
	else:
		attention = False

	# DENSE LAYER 1
	layer_outputs = [layer.output for layer in classifier.layers[len(x_data):-2]]
	dense_model = models.Model(inputs=classifier.input, outputs=layer_outputs)
	dense_model.summary()
	dense1 = dense_model.predict(x_data)[-1]

	# DENSE LAYER 1
	layer_outputs = [layer.output for layer in classifier.layers[len(x_data):-1]]
	dense_model = models.Model(inputs=classifier.input, outputs=layer_outputs)
	dense_model.summary()
	dense2 = dense_model.predict(x_data)[-1]

	# READ PREDICTION SENTENCE BY SENTENCE
	word_nb = 0
	for sentence_nb in range(len(x_data[0])):

		# CSV
		csv = "Source;Target;Weight;Type\n"
		csv2= "ID;Type\n"

		print(sentence_nb , "/" , len(x_data[0]))
		sentence = {}
		sentence["sentence"] = []
		sentence["prediction"] = dense2[sentence_nb].tolist()
		prediction_index = sentence["prediction"].index(max(sentence["prediction"]))

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
			word = {}
			for channel in range(preprocessing.nb_channels):
				#print(channel , "/" , len(tds))

				if not tds:
					tds_value = 0
				else:
					# DECONV BY READING FILTERS)

					# OLD TDS
					#tds_value = sum(tds[-(channel+1)][sentence_nb][i])

					# NEW TDS
					from_i = i*config["EMBEDDING_DIM"]
					to_j = from_i + config["EMBEDDING_DIM"]
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

				# FILL THE WORD VALUES
				channel_name = "channel" + str(channel)
				word[channel_name] = {}

				#print(x_data[channel][sentence_nb].shape)
				index = x_data[channel][sentence_nb][i]

				word[channel_name]["str"] = dictionaries[channel]["index_word"][index]
				if word[channel_name]["str"] == "UK":
					word[channel_name]["str"] = raw_text[word_nb].split("**")[channel]
				word[channel_name]["tds"] = str(tds_value)
				word[channel_name]["attention"] = str(attention_value)
				if config["ENABLE_LIME"]:
					word[channel_name]["lime"] = lime[sentence_nb][word[channel_name]["str"]]
				
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

			sentence["sentence"] += [word]	
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
	return result
