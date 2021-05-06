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

	# Load text
	f = open(text_file, "r")
	lines = f.readlines()

	# ----------------------------
	# BOOTSTRAP TEXT : TOKENIZE + MOPH ANALYSE
	docs = list(preprocessing.nlp.pipe(lines, n_process=-1, batch_size=8))
	bootstrap = {}
	bootstrap["FORME"] = []
	bootstrap["CODE"] = []
	for doc in docs:
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
	for i in range(100):

		# Get next predicted word
		preprocessing.loadSequence(bootstrap, config, concate)

		# FORM prediction
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

		prediction_code = bootstrap["CODE"][i]

		"""
		if i%2==0:
			current_codes = bootstrap["CODE"] + [c[1] for c in concate]
			#print(current_codes)
			matching_list = []
			for sgram in preprocessing.sgram["CODE"]:
				if list(sgram[:-1]) == current_codes[-(len(sgram)-1):]:
					matching_list += [sgram]
			prediction_code = random.choice(matching_list)[-1]
		"""

		current_text = bootstrap["FORME"] + [c[0] for c in concate]

		ttl = 0
		while ttl < 100:
			max_pred = predictions.index(max(predictions))
			prediction = preprocessing.reversedictionary["FORME"][max_pred]
			text = " ".join(current_text + [prediction])
			doc = list(preprocessing.nlp(text))
			code = preprocessing.get_code(doc[-1])			
			predictions[max_pred] = -1
			if code == prediction_code: break
			ttl += 1

		if ttl >= 100:
			matching_list = []
			for sgram in preprocessing.sgram["FORME"]:
				if list(sgram[:-1]) == current_text[-(len(sgram)-1):]:
					matching_list += [sgram]
			prediction = random.choice(matching_list)[-1]

			
		log_data["message"] += prediction + " "
		print(prediction, end=' ', flush=True)
		concate += [(prediction, prediction_code)]

		# WRITE CURRENT TEXT IN A LOG FILE
		try:
			open(log_file, "w").write(json.dumps(log_data))
		except:
			pass

	print("\n")
	return "" #" ".join(bootstrap["FORME"]) + " " + " ".join(concate)