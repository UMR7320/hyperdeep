import numpy as np

from keras import optimizers
from keras import regularizers

from keras import backend as K
from keras.models import Sequential,Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import Reshape
from keras.layers import TimeDistributed
from keras.layers import Activation
from keras.layers import RepeatVector
from keras.layers import Permute
from keras.layers import Lambda
from keras.layers import Conv1D, UpSampling1D, Conv2D, Conv2DTranspose, MaxPooling1D, MaxPooling2D, GlobalMaxPooling1D,  Embedding, Reshape
from keras.layers import Input, Embedding, LSTM, GRU, Bidirectional, Dense
from keras.layers import Lambda
from keras.layers import concatenate
from keras.utils import np_utils
from keras.layers import multiply
from keras.layers import BatchNormalization
from keras.utils import multi_gpu_model

from tensorflow.python.client import device_lib

class CNNModel:

	def get_available_gpus(self):
	    local_device_protos = device_lib.list_local_devices()
	    return [x.name for x in local_device_protos if x.device_type == 'GPU']
	
	def getModel(self, config, weight=None):

		print("-"*20)
		print("CREATE MODEL")
		print("-"*20)
		
		# ---------------------------------------
		# MULTI CHANNELS CONVOLUTION
		# ---------------------------------------
		if config["TG"]:
			nb_channels = 3
		else:
			nb_channels = 1

		inputs = [0]*nb_channels
		embedding = [0]*nb_channels

		config["FILTER_SIZES"] = config["FILTER_SIZES"].split("-")
		
		conv = [0]*nb_channels
		pool = [0]*nb_channels

		for i in range(nb_channels):
			print("CHANNELS ", i)

			# INPUTS
			inputs[i] = Input(shape=(config["SEQUENCE_SIZE"],), dtype='int32')
			print("input", i,  inputs[i].shape)

			# EMBEDDING
			if config["SG"] == -1:
				weights = None
			else:
				weights=[weight[i]]
			embedding[i] = Embedding(
				config["vocab_size"][i],
				config["EMBEDDING_DIM"],
				input_length=config["SEQUENCE_SIZE"],
				weights=weights,
				trainable=config["EMBEDDING_TRAINABLE"]
			)(inputs[i])
			print("embedding", i,  embedding[i].shape)

			embedding[i] = BatchNormalization()(embedding[i])

			last_layer = embedding[i]

			# CONVOLUTIONs
			if config["ENABLE_CONV"]:
				for FILTER_SIZES in config["FILTER_SIZES"]:
					FILTER_SIZES = int(FILTER_SIZES)
					
					conv[i] = Conv1D(filters=config["NB_FILTERS"], strides=1, kernel_size=FILTER_SIZES, padding='same', kernel_initializer='normal', activation='relu')(last_layer)
					print("conv", i,  conv[i].shape)

					pool[i] = MaxPooling1D(pool_size=2, strides=2, padding='same')(conv[i])
					print("pool", i,  pool[i].shape)
					
					last_layer = pool[i]
					
				print("-"*20)

				# DECONVOLUTION
				conv[i] = UpSampling1D(2**len(config["FILTER_SIZES"]))(last_layer)
				conv[i] = Conv1D(filters=config["EMBEDDING_DIM"], strides=1, kernel_size=FILTER_SIZES, padding='same', kernel_initializer='normal', activation='relu')(conv[i])

		# ------------------------------------		
		# APPLY THE MULTI CHANNELS ABSTRACTION
		# ------------------------------------
		if config["ENABLE_CONV"]:
			if config["TG"]:
				merged = concatenate(conv)
			else:
				merged = conv[0]
		else:
			if config["TG"]:
				merged = concatenate(embedding)
			else:
				merged = embedding[0]
		print("merged", merged.shape)

		if config["ENABLE_LSTM"]:

			# --------------------------------
			# Normalization (forme/code/lemme)
			# --------------------------------			
			#merged = BatchNormalization()(merged)

			# ----------
			# LSTM LAYER
			# ----------
			rnn = Bidirectional(GRU(config["LSTM_SIZE"], return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(embedding[0])
			print("rnn :", rnn.shape)

			# ---------------
			# ATTENTION LAYER
			# ---------------
			attention = TimeDistributed(Dense(1, activation='tanh'))(rnn) 
			print("TimeDistributed :", attention.shape)

			# reshape Attention
			attention = Flatten()(attention)
			print("Flatten :", attention.shape)
			
			attention = Activation('softmax')(attention)
			print("Activation :", attention.shape)

			# Pour pouvoir faire la multiplication (scalair/vecteur KERAS)
			# attention = RepeatVector(config["LSTM_SIZE"])(attention) # NORMAL RNN
			attention = RepeatVector(config["LSTM_SIZE"]*2)(attention) # BIDIRECTIONAL RNN
			print("RepeatVector :", attention.shape)
			
			attention = Permute([2, 1])(attention)
			print("Permute :", attention.shape)

			# apply the attention		
			sent_representation = multiply([rnn, attention])
			print("Multiply :", sent_representation.shape)
		
			# -------------
			# DROPOUT LAYER
			# -------------
			flat = Flatten()(sent_representation)
		else:
			flat = Flatten()(merged)
			
		dropout = Dropout(config["DROPOUT_VAL"])(flat)
		print("Dropout :", dropout.shape)

		# ------------------
		# HIDDEN DENSE LAYER
		# ------------------	
		hidden_dense = Dense(config["DENSE_LAYER_SIZE"], kernel_initializer='uniform',activation='relu')(dropout)
		#hidden_dense = Activation('relu')(dropout)

		# -----------------
		# FINAL DENSE LAYER
		# -----------------
		crossentropy = 'categorical_crossentropy'
		output_acivation = 'softmax'

		output = Dense(config["num_classes"],kernel_regularizer=regularizers.l1(0.01))(hidden_dense)
		output = Activation(output_acivation)(output)

		print("output :", output.shape)

		# -----------------
		# COMPILE THE MODEL
		# -----------------
		model = Model(inputs=inputs, outputs=output)
		# For multi-gpu
		if len(self.get_available_gpus()) >= 2:
			model = multi_gpu_model(model, gpus=2)
		op = optimizers.Adam(lr=config["LEARNING_RATE"], beta_1=0.9, beta_2=0.999, epsilon=1e-08)
		model.compile(optimizer=op, loss=crossentropy, metrics=['accuracy'])

		print("-"*20)
		print("MODEL READY")
		print("-"*20)

		print("TRAINING MODEL")
		print(model.summary())

		return model