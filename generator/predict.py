#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
Created on 20 dec. 2020
@author: lauren

t.vanni@unice.fr
'''
import random
import json
from os import environ, system
from termcolor import colored

from tensorflow.keras.models import load_model
from generator.preprocessing import PreProcessing

# ------------------------------
# GENERATE
# ------------------------------
def generate(model_file, text_file, log_file, config):
	print("GENERATE")
	environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	environ["CUDA_VISIBLE_DEVICES"] = ""
	
	corpus_file = model_file.replace("bin/", "data/" )
	print("PREPROCESSING...")

	# GET PREPROCESSING
	preprocessing = PreProcessing(model_file)
	print("PREPROCESSING DONE.")

	# LOG FILE (update results)
	try:
		log_data = json.load(open(log_file, "r"))
	except:
		log_data = {}
	log_data["message"]  = "__GEN__"

	# Get spec
	print("LOAD SPEC")
	spec = json.loads(open(model_file + ".spec", "r").read())
	vocab = list(spec.keys())

	print("-LOAD MODEL-")
	model = load_model(model_file+".lang")
	text = ""
	concate = []

	system("clear");
	preprocessing.loadBootstrap(text_file, config, concate)
	print(colored(preprocessing.X_bootstrap, 'cyan'), end=' ', flush=True)

	print("START GEN...")
	for i in range(100):

		# Get next predicted word
		preprocessing.loadBootstrap(text_file, config, concate)
		predictions = list(model.predict(preprocessing.X)[0])
		max_pred = predictions.index(max(predictions))
		weighted_predictions = [p*10000 for p in predictions]
		#max_pred = random.choices(range(len(weighted_predictions)), weights=weighted_predictions, k=1)[0]
		prediction = preprocessing.reversedictionary[max_pred]
	    
	    # ----------------------------------------------------		
		# Adjust prediction
		#if len(concate) > config["WORD_LENGTH"]:
		"""
			try:
				pred_count = concate.count(prediction)+1
				k = spec[prediction]["k"] + pred_count
				f = spec[prediction]["f"] + pred_count
				t = spec[vocab[0]]["t"] + len(concate)+1
				T = spec[vocab[0]]["T"] + len(concate)+1
				z1 = z_score(spec[prediction]["k"], spec[prediction]["f"], spec[vocab[0]]["t"], spec[vocab[0]]["T"])
				z2 = z_score(k, f, t, T)
				delta = z1 - z2
				#print(prediction, k, f, t, T, z1, z2)
			except:
				delta = 0

			if delta != 0:
			print(prediction, delta)
		"""

		try:
			if spec[prediction]["z"] < 1:
				give_a_chance = [1, 10]
			else:
				give_a_chance = [int(spec[prediction]["z"]), 10]
		except:
			give_a_chance = [5, 5]

		current_text = preprocessing.X_bootstrap.strip().split(" ") + concate
		ttl = 0
		
		# CONDITIONS :
		# ------------
		# NGRAM
		cond1 = (current_text[-2], current_text[-1], prediction) not in preprocessing.sgram
		# AVOID LOOP
		cond2 = prediction in current_text and random.choices([0, 1], weights=give_a_chance, k=1)[0]

		while ttl < 100 and (cond1 or cond2):
			
			#predictions[max_pred] = -1
			#max_pred = predictions.index(max(predictions))	
			if cond1:
				predictions[max_pred] = -1
				max_pred = predictions.index(max(predictions))
			else:
				max_pred = random.choices(range(len(weighted_predictions)), weights=weighted_predictions, k=1)[0]
			prediction = preprocessing.reversedictionary[max_pred]
			
			ttl += 1
			try:
				if spec[prediction]["z"] < 1:
					give_a_chance = [1, 10]
				else:
					give_a_chance = [int(spec[prediction]["z"]), 10]
			except:
				give_a_chance = [5, 5]
			# NGRAM
			cond1 = (current_text[-2], current_text[-1], prediction) not in preprocessing.sgram
			# AVOID LOOP
			cond2 = prediction in current_text and random.choices([0, 1], weights=give_a_chance, k=1)[0]

		# ----------------------------------------------------

		if ttl == 100:
			for gram in preprocessing.lgram:
				if current_text[-2] == gram[0] and current_text[-1] == gram[1]:
					prediction = gram
					break
			concate = concate[:-2] + list(prediction)
			for word in prediction[2:]:
				print(colored(word, 'cyan'), end=' ', flush=True)
				log_data["message"] += word + " "
		else:
			if prediction != "\n" or prediction != concate[-1] or concate[-1] != concate[-2]:
				log_data["message"] += prediction + " "
			print(prediction, end=' ', flush=True)
			concate += [prediction]

		# WRITE CURRENT TEXT IN A LOG FILE
		try:
			open(log_file, "w").write(json.dumps(log_data))
		except:
			pass

	print("\n")
	#print(""-"*50")
	#print(preprocessing.X_bootstrap.strip() + " " + " ".join(concate))

	return preprocessing.X_bootstrap.strip() + " " + " ".join(concate)