#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Created on 16 nov. 2017
@author: laurent.vanni@unice.fr
'''
import os
import numpy as np
import platform

# Plot for log
import matplotlib.pyplot as plt

# Deep learning librairies
from keras.utils import plot_model
from keras.utils import np_utils
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model

# Exaplainer librairies
from analyzer.lime import LimeExplainer
from analyzer.tds import computeTDS

# Model dependencies
from preprocess.general import PreProcessing
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
	plt.savefig(os.path.join('results', "accuracy.png"))
	plt.show()

# ------------------------------
# TRAIN
# ------------------------------
def train(corpus_file, model_file, config):
	
	# preprocess data
	preprocessing = PreProcessing(model_file, config)
	preprocessing.loadData(corpus_file)
	preprocessing.encodeData(forTraining=True)
	
	if config["SG"] != -1:
		preprocessing.loadEmbeddings()
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
	print("TRAIN CLASSIFIER")
	print("-"*65)
	history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=config["NUM_EPOCHS"], batch_size=config["BACH_SIZE"], callbacks=callbacks_list)

	# ------------------------------------
	# GET EMBEDDING MODEL
	print("-"*50)
	print("FINAL EMBEDDING")
	print("-"*50)
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
	scores = model.evaluate(x_val, y_val, verbose=1)

	# ------------------------------------
	# Plot training & validation loss values
	if "plot" in config.keys():
		plot_history(history)

	return scores

# ------------------------------
# PREDICT
# ------------------------------
def predict(text_file, model_file, config):

	# ------------------------------------------
	# Force to use CPU (no need GPU on predict)
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
	os.environ["CUDA_VISIBLE_DEVICES"] = ""

	# ------------------------------------------
	# Preprocess data 
	preprocessing = PreProcessing(model_file, config)
	preprocessing.loadData(text_file)
	preprocessing.encodeData(forTraining=False)

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
			if "plot" in config.keys():
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
				plt.savefig(os.path.join('results', method + ".png"))
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
			# plt.savefig(os.path.join('results', method + ".png"))
			# plt.show()
			if "plot" in config.keys():
				lime_html = lime_explainers[0].as_html()
				open(os.path.join('results', "lime.html"), "w").write(lime_html)
				try:
					print("Trying to open lime.html...")
					if platform.system() == "Windows":
						os.system("start " + os.path.join('results', "lime.html"))
					else:
						os.system("open " + os.path.join('results', "lime.html"))
				except:
					print("Failed.")
					print("Open results/lime.html file manualy to see lime explainer")

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