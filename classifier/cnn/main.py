import random
import numpy as np
import timeit

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
		for line in f.readlines():

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

		# split the data into a training set and a validation set
		indices = np.arange(datas[0].shape[0])
		np.random.shuffle(indices)
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

	def loadEmbeddings(self, model_file, config, create_v = False):
		
		self.embedding_matrix = []

		if not create_v:
			create_vectors(self.corpus_file, model_file, config, nb_channels=len(self.dictionaries))

		for i, dictionary in enumerate(self.dictionaries):
			my_dictionary = dictionary["word_index"]
			embeddings_index = {}
			vectors = open(model_file + "." + str(i) + ".vec"  ,'r')
				
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
	checkpoint = ModelCheckpoint(model_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]
	model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=config["NUM_EPOCHS"], batch_size=config["BACH_SIZE"], callbacks=callbacks_list)

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

	print("----------------------------")
	print("DECONVOLUTION")
	print("----------------------------")
	# GET THE CONVOLUTIONAL LAYERS
	isConvLayer = False
	last_conv_layer = 0
	i = 0
	for layer in classifier.layers:	
		if type(layer) is Activation:
			last_conv_layer = i+1
		i += 1
	layer_outputs = [layer.output for layer in classifier.layers[:last_conv_layer]] 
	deconv_model = models.Model(inputs=classifier.input, outputs=layer_outputs)
	deconv_model.summary()
	deconv = deconv_model.predict(x_data)
	for channel in range(len(x_data)):
		print(deconv[-(channel+1)][0].shape)

	# READ PREDICTION SENTENCE BY SENTENCE
	for sentence_nb in range(len(x_data[channel])):
		sentence = {}
		sentence["sentence"] = ""
		sentence["prediction"] = predictions[sentence_nb].tolist()

		# READ SENTENCE WORD BY WORD
		for i in range(config["SEQUENCE_SIZE"]):
			word = ""
			for channel in range(len(x_data)):
				index = x_data[channel][sentence_nb][i]
				word += dictionaries[channel]["index_word"].get(index, "PAD")

				"""
				if i == 0 or i == config["SEQUENCE_SIZE"]:
					word += "*1"
				else:
				"""
				attention = deconv[-(channel+1)][sentence_nb][i]*1000
				print("Attention:", attention)
				word += "*" + str(attention)

				word += "**"
			word = word[:-1] + "0" # attention...
			sentence["sentence"] += word + " "
		result.append(sentence)

		# ------ DRAW DECONV FACE ------
		"""
		deconv_image = np.zeros( (config["SEQUENCE_SIZE"]*len(x_data), config["EMBEDDING_DIM"], 3), dtype=np.uint8 )
		for channel in range(len(x_data)):
			for y in range(config["SEQUENCE_SIZE"]):
				deconv_value = deconv[channel][sentence_nb][y]
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
