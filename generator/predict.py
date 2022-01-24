#!/usr/bin/python3
# -*- coding: utf-8 -*-
'''
Created on 20 dec. 2020
@author: lauren

t.vanni@unice.fr
'''
import random
import json
import re

from os import environ, system
from termcolor import colored

from tensorflow.keras.models import load_model
from ..generator.preprocessing import PreProcessing

from nltk import ngrams

# ------------------------------
# GENERATE
# ------------------------------
def generate(model_file, bootstrap_raw, result_file, config):

	jalons = [["conclusion", "conclure", "deep", "learning", "conclusion"], 
	["travaux", "remarques", "deep", "learning", "ADT", "semble", "IA", "intertextuels"], 
	["perspective", "TDS", "nouveaux", "observables", "linguistiques"]]
	j=0
	textgen_size = 100

	print("GENERATE", model_file)
	environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	environ["CUDA_VISIBLE_DEVICES"] = ""
	
	corpus_file = model_file.replace("bin/", "data/" )
	print("PREPROCESSING...")

	# GET PREPROCESSING
	preprocessing = PreProcessing(model_file)
	print("PREPROCESSING DONE.")

	# ----------------------------
	# BOOTSTRAP TEXT : TOKENIZE + MOPH ANALYSE
	doc = list(preprocessing.nlp(bootstrap_raw))
	bootstrap = {}
	bootstrap["FORME"] = []
	bootstrap["CODE"] = []
	for token in doc:
		# TOKENIZE
		bootstrap["FORME"] += [token.text]
		bootstrap["CODE"] += [preprocessing.get_code(token)]
	# ----------------------------

	# LOG FILE (update results)
	result_data = json.load(open(result_file, "r"))
	result_data["data"] = bootstrap_raw

	# Get spec
	print("LOAD SPEC")
	spec = json.loads(open(model_file + ".spec", "r").read())
	vocab = list(spec.keys())

	print("-LOAD MODEL-")
	forme_model = load_model(model_file + "_FORME" + ".lang")
	code_model = load_model(model_file + "_CODE" + ".lang")
	text = ""
	concate={"FORME":[], "CODE":[]}

	system("clear");
	print(colored(" ".join(bootstrap["FORME"]), 'cyan'), end=' ', flush=True)


	for i in range(textgen_size):

		print_jalon = False

		# Get next predicted word
		preprocessing.loadSequence(bootstrap, config, concate)

		# FORME prediction
		predictions = list(forme_model.predict(preprocessing.X["FORME"])[0])

		# CODE prediction
		predictions_code = list(code_model.predict(preprocessing.X["CODE"])[0])
		max_pred = predictions_code.index(max(predictions_code))
		prediction_code = preprocessing.indexes["CODE"][max_pred]
		if prediction_code == "SPACE":
			predictions_code[max_pred] = -1
			max_pred = predictions_code.index(max(predictions_code))
			prediction_code = preprocessing.indexes["CODE"][max_pred]
		
		current_text = bootstrap["FORME"] + concate["FORME"] #[c[0] for c in concate]

		# Jalons conditions
		if j < len(jalons):
			if len(current_text) + random.randint(0,int(textgen_size/10)) > (textgen_size/len(jalons))*(j+1):
				if current_text[-1] == ".":
					print("\n")
					result_data["data"] += "\n\n"
					j += 1
			else:
				for p, jalon in enumerate(jalons[j]):
					jalon = jalon.split()
					if tuple(current_text[-2:] + [jalon[-1]]) in preprocessing.sgram["FORME"]:
						if len(jalon) == 1 or jalon[-len(jalon):-1] == current_text[-(len(jalon)-1):]:
							prediction = jalon[-1]
							del jalons[j][p]
							print_jalon = True
							break
		ttl = 0
		while ttl < 100 and not print_jalon:
			max_pred = predictions.index(max(predictions))
			prediction = preprocessing.indexes["FORME"][max_pred]

			text = " ".join(current_text + [prediction])
			doc = list(preprocessing.nlp(text))
			code = preprocessing.get_code(doc[-1])	

			# CONDITIONS :
			# ------------

			# NGRAM
			cond1 = tuple(current_text[-(config["WORD_LENGTH"]-1):] + [prediction]) in preprocessing.lgram["FORME"]
			#cond1 = tuple(current_text[-2:] + [prediction]) in preprocessing.sgram["FORME"]
			#cond1 = code == prediction_code

			# AVOID LOOP
			current_ngram = list(ngrams(current_text, 3))
			try:
				if spec[prediction]["z"] < 1:
					give_a_chance = [1, 10]
				else:
					give_a_chance = [int(spec[prediction]["z"]), 10]
			except:
				give_a_chance = [5, 5]
			cond2 = not tuple(current_text[-2:] + [prediction]) in  current_ngram or (len(prediction) > 4 and random.choices([0, 1], weights=give_a_chance, k=1)[0])
			
			#break
			#if cond1: break
			#if cond2: break
			if cond1 and cond2: break
			
			# Next loop
			predictions[max_pred] = -1
			ttl += 1

		# --------------------------------------------------------
		# NOT PREDICTION MATCH
		# GET RANDOM NGRAM
		if ttl == 100:
			#print("TLL == 100")
			#break
			"""
			concate["FORME"] += ["."]
			result_data["data"] += " ."
			random.shuffle(preprocessing.lgram["FORME"])
			for gram in preprocessing.lgram["FORME"]:
				gram = list(gram)
				if gram[0].istitle():
					prediction = gram
					if any(w in jalons[j] for w in gram):
						break
			concate["FORME"] += prediction
			for word in prediction:
				print(colored(word, 'cyan'), end=' ', flush=True)
				result_data["data"] += " " + word
			"""
			random.shuffle(preprocessing.lgram["FORME"])
			prediction = False
			for gram in preprocessing.lgram["FORME"]:
				if current_text[-3:] == list(gram[:3]):
					prediction = gram
					break
			if prediction:
				#print("add lgram")
				concate["FORME"] = concate["FORME"][:-3] + list(prediction)
				for word in prediction[3:]:
					print(colored(word, 'cyan'), end=' ', flush=True)
					result_data["data"] += " " + word
			else:
				random.shuffle(preprocessing.sgram["FORME"])
				prediction = preprocessing.sgram["FORME"][0]
				for gram in preprocessing.sgram["FORME"]:
					if current_text[-2:] == list(gram[:2]):
						prediction = gram
						break
				concate["FORME"] = concate["FORME"][:-2] + list(prediction)
				#if (prediction == preprocessing.sgram["FORME"][0]):
				#	print("add RANDOM sgram")
				#else:
				#	print("add sgram")
				for word in prediction[2:]:
					print(colored(word, 'cyan'), end=' ', flush=True)
					result_data["data"] += " " + word
		# --------------------------------------------------------
		else:
			# APPEND FORME
			result_data["data"] += " " + prediction
			if print_jalon:
				print(colored(prediction, 'red'), end=' ', flush=True)
			else:
				print(prediction, end=' ', flush=True)
				#print(prediction, end='', flush=True)
				#print(colored(":" + prediction_code, 'yellow'), end=' ', flush=True)
			concate["FORME"] += [prediction]

			# APPEND CODE
			text = " ".join(current_text + [prediction])
			doc = list(preprocessing.nlp(text))
			code = preprocessing.get_code(doc[-1])
			concate["CODE"] += [code]

		# WRITE CURRENT TEXT IN A LOG FILE
		result_data["data"] = reverse_tokenizing(result_data["data"])

		try:
			open(result_file, "w").write(json.dumps(result_data))
		except:
			pass

	print("\n")
	return " ".join(bootstrap["FORME"]) + " ".join(concate["FORME"])

# ------------------------------
# TOOLS
# ------------------------------
def reverse_tokenizing(text):
	text = text.replace("’", "'")
	text = text.replace(" '", "'")
	text = text.replace("' ", "'")
	text = text.replace(" -", "-")
	text = text.replace("- ", "-")
	text = text.replace("*", "")
	text = re.sub(' ([.,;:!?"]) ', r'\1 ', text.strip())
	return text

