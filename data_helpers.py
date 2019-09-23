import pickle
import numpy as np

def tokenize(texts, model_file, create_dictionnary, config):

	if create_dictionnary:
		dictionaries = []
		indexes = [1,1,1]
		for i in range(3):
			dictionary = {}
			dictionary["word_index"] = {}
			dictionary["index_word"] = {}
			dictionary["word_index"]["PAD"] = 0  # Padding
			dictionary["index_word"][0] = "PAD"
			dictionary["word_index"]["UK"] = 1 # Unknown word
			dictionary["index_word"][1] = "UK" 
			dictionaries += [dictionary]
	else:
		with open(model_file + ".index", 'rb') as handle:
			dictionaries = pickle.load(handle)
	datas = []		

	type = ["FORME", "CODE", "LEM"]
	for i, text in texts.items():
		datas += [(np.zeros((len(text), config["SEQUENCE_SIZE"]))).astype('int32')]

		line_number = 0
		for line in text:
			words = line.split()[:config["SEQUENCE_SIZE"]]
			sentence_length = len(words)
			sentence = []
			for word in words:
				if word not in dictionaries[i]["word_index"].keys():
					if create_dictionnary:
						skip_word = False
						for spec in config["Z_SCORE"].values():
							if word.isdigit() or len(word) or (word in spec[type].keys() and spec[type][word]["f"] ==   spec[type][word]["k"]):
								is_specific = True
								break

						if skip_word: # Short words considers has UK
							print("remove: ", word)
							dictionaries[i]["word_index"][word] = dictionary["word_index"]["UK"]
						else:	 
							indexes[i] += 1
							dictionaries[i]["word_index"][word] = indexes[i]
							dictionaries[i]["index_word"][indexes[i]] = word

					else:        
						# FOR UNKNOWN WORDS
						try:
							dictionaries[i]["word_index"][word] = dictionaries[i]["word_index"]["UK"]
						except:
							dictionaries[i]["word_index"][word] = dictionaries[i]["word_index"]["PAD"]
				sentence.append(dictionaries[i]["word_index"][word])

			# COMPLETE WITH PAD IF LENGTH IS < SEQUENCE_SIZE
			if sentence_length < config["SEQUENCE_SIZE"]:
				for j in range(config["SEQUENCE_SIZE"] - sentence_length):
					sentence.append(dictionaries[i]["word_index"]["PAD"])
			
			datas[i][line_number] = sentence
			line_number += 1

	if create_dictionnary:
		with open(model_file + ".index", 'wb') as handle:
			pickle.dump(dictionaries, handle, protocol=pickle.HIGHEST_PROTOCOL)

	return dictionaries, datas