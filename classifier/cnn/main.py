import random
import numpy as np
import timeit

from keras.utils import np_utils
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D

from classifier.cnn import models
from skipgram.skipgram_with_NS import create_vectors, create_tg_vectors
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
					sequence[i] += arg + " "
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
			self.embedding_matrix += [np.zeros((len(my_dictionary) + 1, config["EMBEDDING_DIM"]))]
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
	model, deconv_model = cnn_model.getModel(config=config, weight=preprocessing.embedding_matrix)

	# train model
	x_train, y_train, x_val, y_val = preprocessing.x_train, preprocessing.y_train, preprocessing.x_val, preprocessing.y_val
	checkpoint = ModelCheckpoint(model_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	callbacks_list = [checkpoint]
	model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=config["NUM_EPOCHS"], batch_size=config["BACH_SIZE"], callbacks=callbacks_list)

	"""
	try:
		for deconv in deconv_model:
			# SETUP THE DECONV LAYER WEIGHTS
			for layer in deconv_model.layers:	
				if type(layer) is Conv2D:
					deconv_weights = layer.get_weights()[0]
			deconv_bias = deconv_model.layers[-1].get_weights()[1]
			deconv_model.layers[-1].set_weights([deconv_weights, deconv_bias])
	except:
		print("WARNING: not convolution in this model!")
	"""

	# save deconv model
	deconv_model[0].save(model_file + ".deconv")

	# get score
	model = load_model(model_file)
	scores = model.evaluate(x_val, y_val, verbose=0)
	return scores
	
def predict(text_file, model_file, config, vectors_file):

	result = []

	# GIF ANIMATION
	raw_images = []
	rgb_images = []
	final_images = []

	# preprocess data
	preprocessing = PreProcessing()
	preprocessing.loadData(text_file, model_file, config, create_dictionnary = False)
	#preprocessing.loadEmbeddings(model_file, config, vectors_file)
	
	# load and predict
	x_data = np.concatenate((preprocessing.x_train,preprocessing.x_val), axis=0)
	model = load_model(model_file)
	predictions = model.predict(x_data)
	
	print(predictions)	

	print("----------------------------")
	print("DECONVOLUTION")
	print("----------------------------")

	# load deconv_model
	deconv_model = load_model(model_file + ".deconv")
	
	"""
	try:
		# SETUP THE DECONV LAYER WEIGHTS
		for layer in deconv_model.layers:	
			if type(layer) is Conv2D:
				deconv_weights = layer.get_weights()[0]
		deconv_bias = deconv_model.layers[-1].get_weights()[1]
		deconv_model.layers[-1].set_weights([deconv_weights, deconv_bias])
	except:
		print("WARNING: not convolution in this model!")
	
	# apply deconvolution
	deconv = deconv_model.predict(x_data)
	print("deconvolution", 	deconv.shape)
	"""

	my_dictionary = preprocessing.my_dictionary

	"""
	print("----------------------------")
	print("ATTENTION")
	print("----------------------------")

	# load deconv_model
	attention_model = load_model(model_file + ".attention")	

	# apply deconvolution
	attentions = attention_model.predict(x_data)

	print("attentions", attentions.shape)	

	# Format result (prediction + deconvolution)
	my_dictionary = preprocessing.my_dictionary
	"""
	for sentence_nb in range(len(x_data)):
		sentence = {}
		sentence["sentence"] = ""
		sentence["prediction"] = predictions[sentence_nb].tolist()

		# ------ LEMMATIZED VERSION -------
		if config["TG"]:

			# Normalize deconv values
			j = int(config["EMBEDDING_DIM"]/3)

			#deconv_part = [np.copy(deconv[sentence_nb][:,:j]), np.copy(deconv[sentence_nb][:,j:j+j]), np.copy(deconv[sentence_nb][:,-j:])]
			deconv_values = {}

			# ------------------------
			# Normalize values
			# Very slow operation
			# TODO: Optimize this part
			"""
			for part_nb, part in enumerate(["CODE", "LEMME", "FORME"]):
				ratio = 255 / np.max(deconv_part[part_nb])
				deconv_values[part] = deconv_part[part_nb]
				deconv_values[part] = np.multiply(deconv_values[part], [ratio])
			"""
			# ------------------------
			
			"""
			forme_values = deconv_values["FORME"]
			code_values = deconv_values["CODE"]
			lemme_values = deconv_values["LEMME"]
			"""

			# Create word entry
			for i in range(config["SEQUENCE_SIZE"]):
				index = x_data[sentence_nb][i]
				word = my_dictionary["index_word"].get(index, "PAD")
			
				# READ ATTENTION 
				"""
				if i == 0 or i == len(x_data[sentence_nb])-1: # because shape (?,48,1)
					attention_value = 0
				else:
					try:
						attention_value = attentions[sentence_nb][i-1]
					except: # BUG WITH FILTER_SIZE > 3
						attention_value = 0
				"""

				# WRITE WORD ENTRY
				word_args = word.split("**")

				"""
				# deconvolution forme
				word = word_args[0] + "*" + str(np.sum(forme_values[i]))
				# deconvolution code
				try:
					word += "**" + word_args[1] + "*" + str(np.sum(code_values[i]))
					# deconvolution lemme
					word += "**" + word_args[2] + "*" + str(np.sum(lemme_values[i]))
					# attention
				except:
					pass # PAD VALUE
				word += "*" + str(float(attention_value))
				"""

				word = word_args[0] + "*1**" 
				word += word_args[1] + "*1**"
				word += word_args[2] + "*1"
				# attention
				word += "*1"
				sentence["sentence"] += word + " "
		
		# ------ STANDARD VERSION -------
		else:
			for i in range(config["SEQUENCE_SIZE"]):
				index = x_data[sentence_nb][i]
				word = my_dictionary["index_word"].get(index, "PAD")

				# READ DECONVOLUTION 
				#forme_values = deconv[sentence_nb][i]
				
				# READ ATTENTION 
				"""
				if i == 0 or i == len(x_data[sentence_nb])-1: # because shape (?,48,1)
					attention_value = 0
				else:
					try:
						attention_value = attentions[sentence_nb][i-1]
					except: # BUG WITH FILTER_SIZE > 3
						attention_value = 0
				"""

				# WRITE WORD ENTRY
				# deconvolution
				word = word + "*1"
				# attention
				word += "*1"

				sentence["sentence"] += word + " "

		result.append(sentence)

		# ------ DRAW DECONV FACE ------
		"""
		raw_image = np.zeros( (config["SEQUENCE_SIZE"], config["EMBEDDING_DIM"], 3), dtype=np.uint8 )
		rgb_image = np.zeros( (config["SEQUENCE_SIZE"], config["EMBEDDING_DIM"], 3), dtype=np.uint8 )
		final_image = np.zeros( (config["SEQUENCE_SIZE"], config["EMBEDDING_DIM"], 3), dtype=np.uint8 )
		
		for y in range(config["SEQUENCE_SIZE"]):
			deconv_value = deconv[sentence_nb][y]
			for j in range(int(config["EMBEDDING_DIM"]/3)):
				x = j
				dv = deconv_value[x][0]
				dv = dv*200
				raw_image[y, x] = [dv, dv, dv]
				rgb_image[y, x] = [dv, 0, 0]
				try:
					final_image[y, x] = [0, forme_values[y][j]/2, forme_values[y][j]]
				except:
					final_image[y, x] = [dv, 0, 0]

			for j in range(int(config["EMBEDDING_DIM"]/3)):
				x = j+int(config["EMBEDDING_DIM"]/3)
				dv = deconv_value[x][0]
				dv = dv*200
				raw_image[y, x] = [dv, dv, dv]
				rgb_image[y, x] = [0, dv, 0]
				try:
					final_image[y, x] = [code_values[y][j], code_values[y][j]/2, 0]
				except:
					final_image[y, x] = [0, dv, 0]

			for j in range(int(config["EMBEDDING_DIM"]/3)):
				x = j+int(config["EMBEDDING_DIM"]/3*2)
				dv = deconv_value[x][0]
				dv = dv*200
				raw_image[y, x] = [dv, dv, dv]
				rgb_image[y, x] = [0, 0, dv]
				try:
					final_image[y, x] = [0, lemme_values[y][j], 0]
				except:
					final_image[y, x] = [0, 0, dv]

		img = smp.toimage( raw_image )   # Create a PIL image
		img.save(model_file + "_raw.png")
		raw_images.append(imageio.imread(model_file + "_raw.png"))
		#img.show()                      # View in default viewer

		img = smp.toimage( rgb_image )   # Create a PIL image
		img.save(model_file + "_rgb.png")
		rgb_images.append(imageio.imread(model_file + "_rgb.png"))

		img = smp.toimage( final_image )   # Create a PIL image
		img.save(model_file + "_final.png")
		final_images.append(imageio.imread(model_file + "_final.png"))

	# CREATE THE GIF ANIMATION
	imageio.mimsave(model_file + "_raw.gif", raw_images, duration=0.1)
	imageio.mimsave(model_file + "_rgb.gif", rgb_images, duration=0.1)
	imageio.mimsave(model_file + "_final.gif", final_images, duration=0.1)
	"""

	return result


