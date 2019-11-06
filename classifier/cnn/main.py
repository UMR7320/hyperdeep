import random
import numpy as np
import timeit
import math

from keras.utils import np_utils
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv1D, Conv2D, Conv2DTranspose, Activation
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
		for line in lines:

			if cpt%100 == 0:
				print(cpt, "/", len(lines))

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
				texts[i] = texts.get(i, []) + [sequence[i]]

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
	model, deconv_model = cnn_model.getModel(config=config, weight=preprocessing.embedding_matrix)

	# train model
	x_train, y_train, x_val, y_val = preprocessing.x_train, preprocessing.y_train, preprocessing.x_val, preprocessing.y_val
	checkpoint = ModelCheckpoint(model_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]
	
	model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=config["NUM_EPOCHS"], batch_size=config["BACH_SIZE"], callbacks=callbacks_list)

	# SETUP THE DECONV LAYER WEIGHTS
	for i, deconv in enumerate(deconv_model):
		for layer in deconv.layers:	
			if type(layer) is Conv2D:
				deconv_weights = layer.get_weights()[0]
		deconv_bias = deconv.layers[-1].get_weights()[1]
		deconv.layers[-1].set_weights([deconv_weights, deconv_bias])

		# save deconv model
		deconv.save(model_file + ".deconv" + str(i))

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
	# GET THE CONVOLUTIONAL LAYERS
	isConvLayer = False
	last_conv_layer = 0
	last_attention_layer = 0
	i = 0
	for layer in classifier.layers:	
		if type(layer) is Conv1D:
			last_conv_layer = i+1
		elif type(layer) is Activation:
			last_attention_layer = i+1
		i += 1

	# LAST LAYER
	layer_outputs = [layer.output for layer in classifier.layers[len(x_data):-1]] 
	last_model = models.Model(inputs=classifier.input, outputs=layer_outputs)
	last_model.summary()
	last = last_model.predict(x_data)[-1]
	
	# TDS
	if config["ENABLE_CONV"]:

		# DECONV BY READING FILTERS
		#layer_outputs = [layer.output for layer in classifier.layers[len(x_data):last_conv_layer]] 
		#deconv_model = models.Model(inputs=classifier.input, outputs=layer_outputs)
		#deconv_model.summary()
		#tds = deconv_model.predict(x_data)

		# DECONV BY CONV2DTRANSPOSE
		tds = []
		for channel in range(len(x_data)):
			deconv_model = load_model(model_file + ".deconv" + str(channel))
			tds += [deconv_model.predict(x_data[channel])]
	else:
		tds = False

	# ATTENTION
	if config["ENABLE_LSTM"]:
		layer_outputs2 = [layer.output for layer in classifier.layers[len(x_data):last_attention_layer]] 
		attention_model = models.Model(inputs=classifier.input, outputs=layer_outputs2)
		attention_model.summary()
		attention = attention_model.predict(x_data)
	else:
		attention = False

	# READ PREDICTION SENTENCE BY SENTENCE
	word_nb = 0
	for sentence_nb in range(len(x_data[channel])):
		print(sentence_nb , "/" , len(x_data[channel]))
		sentence = {}
		sentence["sentence"] = []
		sentence["prediction"] = last[sentence_nb].tolist()

		# READ SENTENCE WORD BY WORD
		for i in range(config["SEQUENCE_SIZE"]):

			# GET ATTENTION VALUE (RNN NETWORK)
			"""
			if not attention or i >= config["SEQUENCE_SIZE"]-2:
				attention_value = 0
			else:
			"""
			try:
				attention_value = attention[-1][sentence_nb][i]						# ATTENTION
			except:
				attention_value = 0
			
			# GET TDS VALUES
			word = {}
			for channel in range(len(x_data)):

				# DECONV BY READING FILTERS
				#if not tds or i == 0 or i == config["SEQUENCE_SIZE"]-1:
				#	tds_value = 0
				#else:
				#	tds_value = sum(tds[-(channel+1)][sentence_nb][i-1])			# TDS
				#tds_value = sum(tds[-(channel+1)][sentence_nb][i])
				
				# DECONV BY CONV2DTRANSPOSE
				if not tds:
					tds_value = 0
				else:
					tds_value = sum(tds[channel][sentence_nb][i])[0]

				# FILL THE WORD VALUES
				channel_name = "channel" + str(channel)
				word[channel_name] = {}
				index = x_data[channel][sentence_nb][i]
				word[channel_name]["str"] = dictionaries[channel]["index_word"][index]
				if word[channel_name]["str"] == "UK":
					word[channel_name]["str"] = raw_text[word_nb].split("**")[channel]
				word[channel_name]["tds"] = str(tds_value)
				word[channel_name]["attention"] = str(attention_value)
				if config["ENABLE_LIME"]:
					word[channel_name]["lime"] = lime[sentence_nb][word[channel_name]["str"]]
			
			sentence["sentence"] += [word]

			word_nb += 1
		
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
