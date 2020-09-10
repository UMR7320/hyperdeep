import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords

def tokenize(texts, model_file, createDictionary, config):

	if createDictionary:
		print("CREATE A NEW DICTIONARY")
		dictionaries = []
		indexes = [1,1,1]
		for i in range(3):
			dictionary = {}
			dictionary["word_index"] = {}
			dictionary["index_word"] = {}
			dictionary["word_index"]["PAD"] = 0  # Padding
			dictionary["index_word"][0] = "PAD"
			dictionary["word_index"]["__UK__"] = 1 # Unknown word
			dictionary["index_word"][1] = "__UK__" 
			dictionaries += [dictionary]
	else:
		with open(model_file + ".index", 'rb') as handle:
			print("OPEN EXISTING DICTIONARY:", model_file + ".index")
			dictionaries = pickle.load(handle)
	datas = []		

	type = ["FORME", "CODE", "LEM"]
	text_formes = texts[0]
	if config["TG"]:
		text_codes = texts[1]
	else:
		text_codes = False

	for channel, text in texts.items():
		datas += [(np.zeros((len(text), config["SEQUENCE_SIZE"]))).astype('int32')]	

		line_number = 0
		for i, line in enumerate(text):
			words = line.split()[:config["SEQUENCE_SIZE"]]
			
			words_formes = text_formes[i].split()[:config["SEQUENCE_SIZE"]]
			try:
				words_codes =  text_codes[i].split()[:config["SEQUENCE_SIZE"]]
			except:
				words_codes = False

			sentence_length = len(words)

			sentence = []
			for j, word in enumerate(words):
				if word not in dictionaries[channel]["word_index"].keys():
					if createDictionary:
						# IF WORD IS SKIPED THEN ADD "UK" word
						try:
							#skip_word = words_codes[j] in config["FILTERS"] and channel != 1
							skip_word = channel != 1 and any(s in words_codes[j].split(":") for s in config["FILTERS"])
						except:
							skip_word = False
						if skip_word: 
							dictionaries[channel]["word_index"][word] = dictionary["word_index"]["__UK__"]
						else:	 
							indexes[channel] += 1
							dictionaries[channel]["word_index"][word] = indexes[channel]
							dictionaries[channel]["index_word"][indexes[channel]] = word

					else:        
						# FOR UNKNOWN WORDS
						dictionaries[channel]["word_index"][word] = dictionaries[channel]["word_index"]["__UK__"]

				sentence.append(dictionaries[channel]["word_index"][word])

			# COMPLETE WITH PAD IF LENGTH IS < SEQUENCE_SIZE
			if sentence_length < config["SEQUENCE_SIZE"]:
				for j in range(config["SEQUENCE_SIZE"] - sentence_length):
					sentence.append(dictionaries[channel]["word_index"]["PAD"])
			
			datas[channel][line_number] = sentence
			line_number += 1

	if createDictionary:
		with open(model_file + ".index", 'wb') as handle:
			pickle.dump(dictionaries, handle, protocol=pickle.HIGHEST_PROTOCOL)

	print("VOCABULARY SIZE:", len(dictionaries[0]["index_word"]))

	return dictionaries, datas