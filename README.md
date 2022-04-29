# HYPERDEEP
Hyperdeep is an explainable text classifier model based on convolutionnal neural network. This application provides three different methods to explain the classifier outputs :
1) Lime : the well-known library that allows to interpret any classifier
2) TDS : A convolutionnal feature extraction approach that gives an activation score for each word in the text
3) wTDS : A weighted version of TDS based on Class Activation Map (CAM) algorithm.

It use a standard CNN for text classification, and include one layer of deconvolution for the automatic detection of linguistic marks. This software is integrated on the Hyperbase web platform (http://hyperbase.unice.fr) - a web toolbox for textual data analysis.

# requirements:
The application use python3.6.x
The package decency list is available in requirements.txt
To deploy a virtual environment from it, use the commands:
	$ python -m venv hyperdeep-env
	$ source hyperdeep-env/bin/activate
	$ pip install -r requirements.txt

# HOW TO USE IT
# data
Datas are stored in the data/ folder. The training set should be splited into phrases of fixed length (50 words by default). And each phrase should have a label name at the beginning of the line. A label is written : __LABELNAME__
Hyperdeep provide a multichannel interpretation approach that allows to interpret data using different word representations. To be used, the data need to be encoded as the following example (three channels : 1) form 2) part-of-speech 3) lemma):
__Descarte__ Je**PRON**je pense**VERB**penser donc**KON**donc je**PRON**je suis**VERB**être
Hyperdeep is distributed with an example of corpus named Campagne2017 (in data/Campagne2017). This is a french corpus of the 2017 presidential election in france. There are 5 main candidates encoded with the labels (Melenchon, Hamon, Macron, Fillon, LePen).

# Train word2vec
You can train a skipgram model (word2vec) by using the command :
	$ python hyperdeep.py word2vec -input data/Campagne2017 -output data/Campagne2017
This command will create a file named Campagne2017.word2vec0 in the folder bin (create the folder if needed). This txt is the vectors file of the training data.

# Test word2vec
the skipgram model can be tested by using the command (example with the word France) :
	$ python hyperdeep.py nn bin/Campagne2017.word2vec0 France

# Train classifier
To train the classifier:
	$ python hyperdeep.py train -input data/demo/Campagne2017.txt -output bin/Campagne2017
The command will create bin/Campagne2017 folder containing the trained model.

# Test the classifier
Then you can make predictions on new text. There is an example in bin/Campagne2017.test. It's a discourse of E. Macron as french president on 31th december 2017. Hyperdeep will split the discourse in fixed length phrases and should predict most of the phrase a E. Macron
	$ python hyperdeep.py predict bin/Campagne2017 data/demo/Campagne2017.test.txt

# Explain classifier
The predict command line will create a result file in the folder result/ (create the folder if needed). This file is a json format file where you can find the activation score for each selected method. To select a method, use optional flags with the command predict:
	$ python hyperdeep.py predict bin/Campagne2017 data/demo/Campagne2017.test.txt [-lime] [-tds] [-wtds]
-lime : correspond to the lime application described in https://arxiv.org/pdf/1602.04938.pdf
-tds : correspond to the Text Deconvolution Saliency (TDS) calcul described in https://hal.archives-ouvertes.fr/hal-01804310 (Sum of the feature map for each word). But the architecture doesn't use a Conv2DTranspose lmayer anymore (see wtds).
-wtds : correspond to the last implementation of an example based on convolutionnal features. It uses a Class Activation Map algorithm as describe in https://tel.archives-ouvertes.fr/tel-03621264

# CONFIGURATIONN FILE
The application use a config file named config.json (in the root directory). The architecture can be managed using the config file with several parameters (Note: the config can be used either to train word2vec or to train classifier). The main parameters are:

- nb_channels : The number of channels used to encode the data

- SEQUENCE_SIZE: Size of the text (use to split input data in samples)

- EMBEDDING_DIM: The size of the embedding

- MIN_COUNT : Correspond to the gensim.models.Word2Vec min_count parameter
- SG : Correspond to the gensim.models.Word2Vec sg parameter
- WINDOW_SIZE: Correspond to the gensim.models.Word2Vec window parameter
- EMBEDDING_TRAINABLE: Boolean value that make the embedding trainable

- ENABLE_CONV : Boolean value that enables convolutional layer
- ENABLE_LSTM : Boolean value that enables LSTM layer

- NB_FILTERS : Number of filters use in convolutional layers
- FILTER_SIZES : Array of size of convolutional kernel (allow multiple hierarchical convolutions). 

- DENSE_LAYER_SIZE : Hidden layer size

- LSTM_SIZE : LSTM size

- DROPOUT_VAL : DropOut size
- NUM_EPOCHS : Number of epoch
- BACH_SIZE : Batch size
- VALIDATION_SPLIT : Size of validation
- TESTING_SPLIT : Size of test
- LEARNING_RATE : learning rate
- SOFTMAX_BREAKDOWN : Boolean value that enables softmax breakdown

- CLASSES : List of labels used to train the model
