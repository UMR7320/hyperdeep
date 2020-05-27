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
		if createDictionary:
			np.random.shuffle(indices)
		if getLabels:
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
				for channel in range(self.nb_channels):
					entry = []
					for i, word in enumerate(t.split(" ")):
						if word == "": # LIME word removing algo
							entry += [1] # ==> replace to __UK__ word see data_helpers
						else:
							idx = word.split("**")[channel]
							entry += [self.dictionaries[channel]["word_index"].get(word, 0)]
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

		P = self.model.predict(X)
		return P
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
	
def predict(text_file, model_file, config, preprocessing=False):

	test_dataset = False

	result = []

	# GIF ANIMATION
	deconv_images = []

	# preprocess data 
	if not preprocessing: 
		preprocessing = PreProcessing()
		preprocessing.loadData(text_file, model_file, config, getLabels=False, createDictionary=False)
		x_data = []
		for channel in range(len(preprocessing.x_train)):
			x_data += [np.concatenate((preprocessing.x_val[channel],preprocessing.x_test[channel],preprocessing.x_train[channel]), axis=0)]
		classifier = load_model(model_file)
	
	else: # ONLY FOR TEST DATASET
		x_data = preprocessing.x_data
		classifier = preprocessing.classifier
		test_dataset = True

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
			#lime_html = open("lime.html", "w")
			#lime_html.write(exp.as_html())
			#print(exp.available_labels())
			#print ('\n'.join(map(str, exp.as_list(label=predicted_label))))
		lime_list = exp.as_list(label=predicted_label)
		lime = {}
		for e in lime_list:
			lime[e[0]] = e[1]

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

	# TDS LAYERS
	if config["ENABLE_CONV"]:
		layer_outputs = [layer.output for layer in classifier.layers[len(x_data):conv_layers[-1]]] 
		deconv_model = models.Model(inputs=classifier.input, outputs=layer_outputs)
		#print("DECONVOLUTION summary:")
		#deconv_model.summary()
		t0 = time.time()
		tds = deconv_model.predict(x_data)#[-1]
		#print("tds predict time", time.time() - t0)
	else:
		tds = False

	if test_dataset:
		layer_outputs = [layer.output for layer in classifier.layers] # STANDARD OUTPUT
	else:
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

		#print(sentence_nb , "/" , len(x_data[0]))

		total = {}	
		
		for i in range(len(x_data[0][sentence_nb])):
			#print(i , "/" , len(tds[-1][sentence_nb]))
			
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
					#tds_value = [tds_value, tds_value]

					# NEW TDS
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
					

				#print(x_data[channel][sentence_nb].shape)
				index = x_data[channel][sentence_nb][i]

				word_str = dictionaries[channel]["index_word"][index]

				if config["ENABLE_LIME"]:
					lime_word_str = ""
					for k in range(preprocessing.nb_channels):
						 idx = x_data[k][sentence_nb][i]
						 lime_word_str += dictionaries[k]["index_word"][idx] + "**"
					tds_value += [lime[lime_word_str.strip("**")]]

				if word_str == "__UK__":
					word_str = preprocessing.raw_text[word_nb].split("**")[channel]
				word += [{word_str : tds_value}]

			sentence[0] += [word]	
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

	for i, sentence in enumerate(x_data[0]):
		if (i==10): break
		sentence_to_String = ""
		for j, w in enumerate(sentence):
			for channel in range(3):
				word = x_data[channel][i][j]
				sentence_to_String += preprocessing.dictionaries[channel]["index_word"][word] + "**"
			sentence_to_String = sentence_to_String.strip("**") + " "

		print(sentence_to_String, y_data[i])
		print("-"*50)

	scores = classifier.evaluate(x_data, y_data, verbose=1)
	print(scores)
	return scores
	results = classifier.predict(x_data)

	nb_feature = 50
	nb_sample = 500
	nb_sample = int(nb_sample/len(config["CLASSES"]))

	predictions_by_classe = {}
	for i, p in enumerate(results):
		classe = np.argmax(y_data[i])
		predictions_by_classe[classe] = predictions_by_classe.get(classe, {})
		predictions_by_classe[classe][i] = p[classe]
	sorted_predictions_by_classe = {}
	for classe, predictions in predictions_by_classe.items():
		sorted_predictions_by_classe[classe] = sorted(predictions_by_classe[classe].items(), key=operator.itemgetter(1), reverse=True)
		sorted_predictions_by_classe[classe] = sorted_predictions_by_classe[classe][:nb_sample]

	try:
		os.remove(model_file+"_lime.csv")
		os.remove(model_file+"_z.csv")
		os.remove(model_file+"_tds.csv")
		os.remove(model_file+"_logit.csv")
	except:
		pass

	# GET SPEC
	spec = json.load(open(model_file + ".spec", "r"))

	# GET LOGIT
	logit_file = open(model_file + "_feat.csv", "r").readlines()
	classes = logit_file[0].split(",")
	logit = {}
	for line in logit_file[1:]:
		args = line.split(",")
		for c, arg in enumerate(args[1:]):
			logit[classes[c+1]] = logit.get(classes[c+1], {})
			logit[classes[c+1]][arg] = args[0]

	for classe, sorted_predictions in sorted_predictions_by_classe.items():

		sample_id = 0
		for p in sorted_predictions:

			if sample_id%20 == 0:
				sess = get_session()
				clear_session()
				sess.close()
				sess = get_session()

				try:
					del preprocessing.classifier # this is from global space - change this as you need
					del tds
					del lime_csv
					del z_csv
					del tds_csv
				except:
					pass

				print(gc.collect()) # if it's done something you should see a number being outputted

				# use the same config as you used to create the session
				tensorflow_config = tensorflow.ConfigProto()
				tensorflow_config.gpu_options.per_process_gpu_memory_fraction = 1
				tensorflow_config.gpu_options.visible_device_list = "0"
				set_session(tensorflow.Session(config=tensorflow_config))

				tracker.print_diff()

				preprocessing.classifier = load_model(model_file)
			
			i = p[0]

			t0 = time.time()

			# GET SENTENCE TO TEST
			preprocessing.x_data = []
			for channel in range(preprocessing.nb_channels):
				preprocessing.x_data += [np.array(x_data[channel][i]).reshape(1, config["SEQUENCE_SIZE"])]

			# GET DEFAUT PREDICTION
			config["ENABLE_LIME"] = True
			tds = predict(corpus_file, model_file, config, preprocessing)		
			predicted_class = tds[0][1].index(max(tds[0][1]))
			predicted_score = tds[0][1][predicted_class]

			#print("t1", time.time() - t0)
			#t0 = time.time()

			sentence=[]
			sentence_to_String = ""
			for j, word in enumerate(tds[0][0]):
				word_str = next(iter(word[0]))
				for channel in range(preprocessing.nb_channels)[1:]:
					word_str += "**" + next(iter(word[channel]))
				sentence += [word_str]
				sentence_to_String += next(iter(word[0])) + " "
			print(sentence_to_String)
			print("PREDICTED CLASSE:", config["CLASSES"][predicted_class], predicted_score)
			
			config["ENABLE_LIME"] = False
			current_processing_data = preprocessing.x_data

			#print("t1", time.time() - t0)
			#t0 = time.time()

			# TEST LIME
			"""
			print("-"*50)
			print("LIME")
			print("-"*50)
			lime_csv = open(model_file+"_lime.csv", "a+")
			lime_csv.write(config["CLASSES"][classe]+"\t"+str(predicted_score)+"\t")
			lime_dic = {}

			lime = []
			for j, word in enumerate(tds[0][0]):
				word_str = next(iter(word[0]))
				if preprocessing.dictionaries[0]["word_index"][word_str] == 1: continue
				lime_value = word[0][word_str][-1]
				for channel in range(preprocessing.nb_channels)[1:]:
					word_str += "**" + next(iter(word[channel]))
				lime_dic[word_str] = lime_value
			lime_dic = sorted(lime_dic.items(), key=operator.itemgetter(1), reverse=True)

			#print("t3", time.time() - t0)
			#t0 = time.time()

			matched_cpt = 0
			for id_feature in range(nb_feature):
				try:
					lime_entry = lime_dic[id_feature]
				except:
					lime_entry = lime_dic[-1]

				#print("t41", time.time() - t0)
				#t0 = time.time()

				words_ids = []
				for channel, arg in enumerate(lime_entry[0].split("**")):
					words_ids += [preprocessing.dictionaries[channel]["word_index"][arg]]
				
				word_str = lime_entry[0]
				entry = []
				test_sentence = []

				X = []
				for channel in range(preprocessing.nb_channels):
					X += [[]]

				#print("t42", time.time() - t0)
				#t0 = time.time()

				for channel in range(preprocessing.nb_channels):
					entry = []
					for i in range(config["SEQUENCE_SIZE"]):
						match = True
						for channel_tmp in range(preprocessing.nb_channels):
							if preprocessing.x_data[channel_tmp][0][i] != words_ids[channel_tmp]:
								match = False
								break
						if match:
							entry += [1] # ==> replace to __UK__ word see data_helpers
							matched_cpt += 1
						else:
							entry += [preprocessing.x_data[channel][0][i]]
					X[channel] += [entry]

				for channel in range(preprocessing.nb_channels):
					X[channel] = np.asarray(X[channel])

				#print(preprocessing.x_data)
				#print(X)
				preprocessing.x_data = X

				#print(preprocessing.x_data)
				print("REMOVING:", lime_entry[0])

				if (id_feature+1)%5 == 0:
					results = predict(corpus_file, model_file, config, preprocessing)
					current_score = results[0][1][predicted_class]
					print("ACCURACY:", current_score, "NB_WORD:", matched_cpt)
					lime_csv.write(str(current_score)+"\t"+str(matched_cpt)+"\t")
					#matched_cpt = 0

			lime_csv.write('\n')
			lime_csv.close()

			# TEST Z-SCORE
			print("-"*50)
			print("Z-SCORE")
			z_csv = open(model_file+"_z.csv", "a+")
			z_csv.write(config["CLASSES"][classe]+"\t"+str(predicted_score)+"\t")
			preprocessing.x_data = current_processing_data

			# PREPARE SPEC
			predicted_spec = spec[config["CLASSES"][predicted_class]]
			z_list = {}
			for channel, spec_type in enumerate(["FORME", "CODE", "LEM"]):
				for word in sentence:
					try:
						word = word.split("**")[channel]
						if channel:
							word = spec_type + ":" + word
						z_list[str(channel) + "_" + word] = predicted_spec[spec_type][word]["z"]
					except:
						print("NO SPEC FOR", spec_type + ":" + word)
						pass
			z_list = sorted(z_list.items(), key=operator.itemgetter(1), reverse=True)

			matched_cpt = 0
			for id_feature in range(nb_feature):
				try:
					spec_entry = z_list[id_feature]
				except:
					spec_entry = z_list[-1]

				X = []
				for channel in range(preprocessing.nb_channels):
					X += [[]]

				word_channel = int(spec_entry[0].split("_")[0])
				word_str = "_".join(spec_entry[0].split("_")[1:])
				if word_channel:
					word_str = ":".join(word_str.split(":")[1:])
				word_id = preprocessing.dictionaries[word_channel]["word_index"][word_str]

				entry = []
				test_sentence = []

				for channel in range(preprocessing.nb_channels):
					entry = []
					for i in range(config["SEQUENCE_SIZE"]):
						#print(channel, word_channel, preprocessing.x_data[channel][0][i], word_id)
						if channel == word_channel and preprocessing.x_data[channel][0][i] == word_id:
							entry += [1] # ==> replace to __UK__ word see data_helpers
							matched_cpt += 1
						else:
							entry += [preprocessing.x_data[channel][0][i]]
						test_sentence += [preprocessing.dictionaries[channel]["index_word"][entry[-1]]]
					X[channel] += [entry]

				print("REMOVING:", spec_entry[0])

				for channel in range(preprocessing.nb_channels):
					X[channel] = np.asarray(X[channel])
				preprocessing.x_data = X
				
				
				if (id_feature+1)%5 == 0:			
					results = predict(corpus_file, model_file, config, preprocessing)
					current_score = results[0][1][predicted_class]
					print("ACCURACY:", current_score, "NB_WORD:", matched_cpt)
					z_csv.write(str(current_score)+"\t"+str(matched_cpt)+"\t")
					#matched_cpt = 0

			z_csv.write('\n')
			z_csv.close()

			# TEST TDS
			print("-"*50)
			print("TDS")
			tds_csv = open(model_file+"_tds.csv", "a+")
			
			preprocessing.x_data = current_processing_data

			tds_csv.write(config["CLASSES"][classe]+"\t"+str(predicted_score)+"\t")
			tds_list = {}
			for channel in range(preprocessing.nb_channels):
				for j, word in enumerate(tds[0][0]):
					current_tds = word[channel][next(iter(word[channel]))][predicted_class]
					
					try:
						word_prev = tds[0][0][j-1]
						prev_tds = word_prev[0][next(iter(word_prev[channel]))][predicted_class]
					except:
						prev_tds = 0
					try:
						word_next = tds[0][0][j+1]
						next_tds = word_next[0][next(iter(word_next[channel]))][predicted_class]
					except:
						next_tds = 0
					if (current_tds > prev_tds and current_tds > next_tds): # TDS Peak
						j = str(channel) + "_" + str(j)
						tds_list[j] = current_tds
			tds_list = sorted(tds_list.items(), key=operator.itemgetter(1), reverse=True)

			matched_cpt = 0
			#for id_feature, tds_entry in enumerate(tds_list[:nb_feature]):
			for id_feature in range(nb_feature):
				try:
					tds_entry = tds_list[id_feature]
				except:
					tds_entry = tds_list[-1]
				X = []
				for channel in range(preprocessing.nb_channels):
					X += [[]]

				word_channel = int(tds_entry[0].split("_")[0])
				word_position = int(tds_entry[0].split("_")[1])
				word_str = ""

				entry = []
				test_sentence = []

				for channel in range(preprocessing.nb_channels):
					entry = []
					for i in range(config["SEQUENCE_SIZE"]):
						current_tds = tds[0][0][i][channel][next(iter(tds[0][0][i][channel]))][predicted_class]
						e = preprocessing.x_data[channel][0][i]
						if e != 1 and i >= word_position-2 and i <= word_position+2 and id_feature < len(tds_list)-1 and current_tds >= tds_list[id_feature+1][1]:
							entry += [1] # ==> replace to __UK__ word see data_helpers
							matched_cpt += 1
							word_str += str(channel)+ "_" + preprocessing.dictionaries[channel]["index_word"][e] + " "
						else:
							entry += [e]
						test_sentence += [preprocessing.dictionaries[channel]["index_word"][entry[-1]]]
					X[channel] += [entry]

				for channel in range(preprocessing.nb_channels):
					X[channel] = np.asarray(X[channel])
				
				preprocessing.x_data = X

				#print(preprocessing.x_data)
				print("REMOVING:", word_str)

				if (id_feature+1)%5 == 0:
					results = predict(corpus_file, model_file, config, preprocessing)
					current_score = results[0][1][predicted_class]
					print("ACCURACY:", current_score, "NB_WORD:", matched_cpt)
					tds_csv.write(str(current_score)+"\t"+str(matched_cpt)+"\t")
					#matched_cpt = 0

			tds_csv.write('\n')
			tds_csv.close()

			# TEST LOGIT
			print("-"*50)
			print("LOGIT")
			logit_csv = open(model_file+"_logit.csv", "a+")
			
			preprocessing.x_data = current_processing_data

			logit_csv.write(config["CLASSES"][classe]+"\t"+str(predicted_score)+"\t")
			
			logit_csv_list = []
			
			predicted_logit = logit[config["CLASSES"][predicted_class]]
			#print(predicted_logit)
			logit_list = {}
			logit_type = ["FORM_", "POS_", "LEM_"]
			for word in sentence:
				try:
					word = word.split("**")
					for channel, word_arg in enumerate(word):
						word_logit = logit_type[channel] + word_arg
						logit_list[str(channel) + "_" + word_arg] = predicted_logit[word_logit]
				except:
					#print("NO LOGIT FOR", logit_type[channel] + word_arg)
					pass
			logit_list = sorted(logit_list.items(), key=operator.itemgetter(1), reverse=True)


			matched_cpt = 0
			for id_feature in range(nb_feature):
				try:
					logit_entry = logit_list[id_feature]
				except:
					logit_entry = logit_list[-1]

				X = []
				for channel in range(preprocessing.nb_channels):
					X += [[]]

				word_channel = int(logit_entry[0].split("_")[0])
				word_str = "_".join(logit_entry[0].split("_")[1:])
				word_id = preprocessing.dictionaries[word_channel]["word_index"][word_str]

				entry = []
				test_sentence = []

				for channel in range(preprocessing.nb_channels):
					entry = []
					for i in range(config["SEQUENCE_SIZE"]):
						#print(channel, word_channel, preprocessing.x_data[channel][0][i], word_id)
						if channel == word_channel and preprocessing.x_data[channel][0][i] == word_id:
							entry += [0] # ==> replace to __UK__ word see data_helpers
							matched_cpt += 1
						else:
							entry += [preprocessing.x_data[channel][0][i]]
						test_sentence += [preprocessing.dictionaries[channel]["index_word"][entry[-1]]]
					X[channel] += [entry]

				print("REMOVING:", logit_entry[0])

				for channel in range(preprocessing.nb_channels):
					X[channel] = np.asarray(X[channel])
				preprocessing.x_data = X
				
				
				if (id_feature+1)%5 == 0:			
					results = predict(corpus_file, model_file, config, preprocessing)
					current_score = results[0][1][predicted_class]
					print("ACCURACY:", current_score, "NB_WORD:", matched_cpt)
					logit_csv.write(str(current_score)+"\t"+str(matched_cpt)+"\t")
					#matched_cpt = 0

			logit_csv.write('\n')
			logit_csv.close()
			"""

			print(config["CLASSES"][classe], sample_id, "/", nb_sample, time.time() - t0)
			print("-"*50)
			sample_id += 1


	return scores