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

from nltk import ngrams

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

	# Load text
	f = open(text_file, "r")

	# ----------------------------
	# BOOTSTRAP TEXT : TOKENIZE + MOPH ANALYSE
	doc = list(preprocessing.nlp(f.read().strip()))
	bootstrap = {}
	bootstrap["FORME"] = []
	bootstrap["CODE"] = []
	for token in doc:
		# TOKENIZE
		bootstrap["FORME"] += [token.text]
		bootstrap["CODE"] += [preprocessing.get_code(token)]
	# ----------------------------

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
	forme_model = load_model(model_file + "_FORME" + ".lang")
	code_model = load_model(model_file + "_CODE" + ".lang")
	text = ""
	concate = []

	system("clear");
	print(colored(" ".join(bootstrap["FORME"]), 'cyan'), end=' ', flush=True)


	print("START GEN...")
	for i in range(200):

		# Get next predicted word
		preprocessing.loadSequence(bootstrap, config, concate)

		# FORME prediction
		predictions = list(forme_model.predict(preprocessing.X["FORME"])[0])

		# CODE prediction
		"""
		predictions_code = list(code_model.predict(preprocessing.X["CODE"])[0])
		max_pred = predictions_code.index(max(predictions_code))
		prediction_code = preprocessing.reversedictionary["CODE"][max_pred]
		if prediction_code == "SPACE":
			predictions_code[max_pred] = -1
			max_pred = predictions_code.index(max(predictions_code))
			prediction_code = preprocessing.reversedictionary["CODE"][max_pred]
		"""
		
		current_text = bootstrap["FORME"] + concate #[c[0] for c in concate]

		ttl = 0
		while ttl < 100:
			max_pred = predictions.index(max(predictions))
			prediction = preprocessing.reversedictionary["FORME"][max_pred]
			#text = " ".join(current_text + [prediction])
			#doc = list(preprocessing.nlp(text))
			#code = preprocessing.get_code(doc[-1])			

			# CONDITIONS :
			# ------------
			# NGRAM
			cond1 = tuple(current_text[-(config["WORD_LENGTH"]-1):] + [prediction]) in preprocessing.lgram["FORME"]
			# AVOID LOOP
			#current_ngram = list(ngrams(current_text, 3))
			"""
			try:
				if spec[prediction]["z"] < 1:
					give_a_chance = [1, 10]
				else:
					give_a_chance = [int(spec[prediction]["z"]), 10]
			except:
				give_a_chance = [5, 5]
			"""
			cond2 = True#not tuple(current_text[-2:] + [prediction]) in  current_ngram or (len(prediction) > 4 and random.choices([0, 1], weights=give_a_chance, k=1)[0])

			if cond1 and cond2: break
			
			# Next loop
			predictions[max_pred] = -1
			ttl += 1

		if ttl == 100:
			random.shuffle(preprocessing.lgram["FORME"])
			for gram in preprocessing.lgram["FORME"]:
				if current_text[-2] == gram[0] and current_text[-1] == gram[1]:
					prediction = gram
					break
			concate = concate[:-2] + list(prediction)
			for word in prediction[2:]:
				print(colored(word, 'cyan'), end=' ', flush=True)
				log_data["message"] += word + " "
		else:
			log_data["message"] += prediction + " "
			print(prediction, end=' ', flush=True)
			concate += [prediction]


		# WRITE CURRENT TEXT IN A LOG FILE
		try:
			open(log_file, "w").write(json.dumps(log_data))
		except:
			pass

	print("\n")
	return " ".join(bootstrap["FORME"]) + " ".join(concate)