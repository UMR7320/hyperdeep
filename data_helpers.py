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
	text_formes = texts[0]
	if config["TG"]:
		text_codes = texts[1]
		try:
			config["FILTERS"] = config["FILTERS"].split()
		except:
			pass
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
					if create_dictionnary:
						
						# FILTERS
						# not a number and len > 1
						skip_word = word.isdigit() or len(word) == 1
						try:
							skip_word = skip_word or (words_codes[j] in config["FILTERS"] and channel != 1)
						except:
							pass

						if not skip_word: # f > k+2%
							for spec in config["Z_SCORE"].values():
								try:
									test_k = spec[type[channel]][word]["k"] + (spec[type[channel]][word]["k"]*0.1)
									if spec[type[channel]][word]["f"] < test_k or spec[type[channel]][word]["f"] < 10:
										skip_word = True
										break
								except :
									pass
						# IF WORD IS SKIPED THEN ADD "UK" word
						if skip_word: 
							dictionaries[channel]["word_index"][word] = dictionary["word_index"]["UK"]
						else:	 
							indexes[channel] += 1
							dictionaries[channel]["word_index"][word] = indexes[channel]
							dictionaries[channel]["index_word"][indexes[channel]] = word

					else:        
						# FOR UNKNOWN WORDS
						try:
							dictionaries[channel]["word_index"][word] = dictionaries[channel]["word_index"]["UK"]
						except:
							dictionaries[channel]["word_index"][word] = dictionaries[channel]["word_index"]["PAD"]
				sentence.append(dictionaries[channel]["word_index"][word])

			# COMPLETE WITH PAD IF LENGTH IS < SEQUENCE_SIZE
			if sentence_length < config["SEQUENCE_SIZE"]:
				for j in range(config["SEQUENCE_SIZE"] - sentence_length):
					sentence.append(dictionaries[channel]["word_index"]["PAD"])
			
			datas[channel][line_number] = sentence
			line_number += 1

	if create_dictionnary:
		with open(model_file + ".index", 'wb') as handle:
			pickle.dump(dictionaries, handle, protocol=pickle.HIGHEST_PROTOCOL)

	print("VOCABULARY SIZE:", dictionaries[0]["index_word"])

	return dictionaries, datas