import numpy as np

from keras import optimizers
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

class CNNModel:
	
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
		reshape = [0]*nb_channels
		conv = [0]*nb_channels
		deconv = [0]*nb_channels
		deconv_model = [0]*nb_channels
		conv_representation = [0]*nb_channels
		pool = [0]*nb_channels
		flat = [0]*nb_channels

		lstm = [0]*nb_channels
		attention = [0]*nb_channels
		sent_representation = [0]*nb_channels

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
				trainable=True
			)(inputs[i])
			print("embedding", i,  embedding[i].shape)

			# RESHAPE
			reshape[i] = Reshape((config["SEQUENCE_SIZE"], config["EMBEDDING_DIM"], 1))(embedding[i])
			print("reshape", i,  reshape[i].shape)

			# CONVOLUTION
			#conv[i] = Conv1D(filters=config["NB_FILTERS"], strides=1, kernel_size=config["FILTER_SIZES"], padding='valid', kernel_initializer='normal', activation='relu')(embedding[i])
			conv[i] = Conv2D(filters=config["NB_FILTERS"], kernel_size=(config["FILTER_SIZES"], config["EMBEDDING_DIM"]), strides=1, padding='valid', kernel_initializer='normal', activation='relu')(reshape[i])
			print("conv", i,  conv[i].shape)

			# MAXPOOLING
			#pool[i] = MaxPooling1D(pool_size=config["SEQUENCE_SIZE"]-2, strides=None, padding='valid')(conv[i])
			#pool[i] = MaxPooling2D(pool_size=(config["SEQUENCE_SIZE"] - config["FILTER_SIZES"] + 1, 1), strides=(1, config["EMBEDDING_DIM"]), padding='valid', data_format='channels_last')(deconv[i])
			#print("pool", i,  pool[i].shape)

			# RESHAPE
			#reshape[i] = Reshape((config["SEQUENCE_SIZE"], 1, config["EMBEDDING_DIM"]))(embedding[i])
			#print("reshape", i,  reshape[i].shape)

			# DECONVOLUTION
			#deconv[i] = UpSampling1D(size=config["SEQUENCE_SIZE"]+2)(pool[i])
			#deconv[i] = Conv1D(filters=config["NB_FILTERS"], kernel_size=config["FILTER_SIZES"], padding='valid', kernel_initializer='normal', activation='relu')(deconv[i])

			deconv[i] = Conv2DTranspose(1, (config["FILTER_SIZES"], config["EMBEDDING_DIM"]), padding='valid', kernel_initializer='normal', activation='relu', data_format='channels_last')(conv[i])
			print("deconv", i,  deconv[i].shape)

			# MAXPOOLING
			#pool[i] = MaxPooling2D(pool_size=(config["SEQUENCE_SIZE"], 1), strides=(1, config["EMBEDDING_DIM"]), padding='valid', data_format='channels_last')(deconv[i])
			#print("pool", i,  pool[i].shape)

			# FLATTEN
			#flat[i] = Flatten()(pool[i])
			#print("flat", i,  flat[i].shape)

			# SUM = SENT REPRESENTATION
			conv_representation[i] = Lambda(lambda xin: K.sum(xin, axis=3))(deconv[i])
			print("Lambda :", i, conv_representation[i].shape)

			# ----------
			# LSTM LAYER
			# ----------
			lstm[i] = Bidirectional(GRU(config["LSTM_SIZE"], return_sequences=True))(conv_representation[i])
			print("lstm :", lstm[i].shape)

			# ---------------
			# ATTENTION LAYER
			# ---------------
			attention[i] = TimeDistributed(Dense(1, activation='tanh'))(lstm[i]) 
			print("TimeDistributed :", attention[i].shape)

			# reshape Attention
			attention[i] = Flatten()(attention[i])
			print("Flatten :", attention[i].shape)
			
			attention[i] = Activation('softmax')(attention[i])
			print("Activation :", attention[i].shape)

			# Observe attention here
			#attention_model = Model(inputs=inputs, outputs=attention)

			# Pour pouvoir faire la multiplication (scalair/vecteur KERAS)
			attention[i] = RepeatVector(config["LSTM_SIZE"]*2)(attention[i])
			print("RepeatVector :", attention[i].shape)
			
			attention[i] = Permute([2, 1])(attention[i])
			print("Permute :", attention[i].shape)

			# apply the attention		
			sent_representation[i] = multiply([lstm[i], attention[i]])
			print("Multiply :", sent_representation[i].shape)
			
			#sent_representation[i] = Lambda(lambda xin: K.sum(xin, axis=2))(sent_representation[i])
			#print("Lambda :", sent_representation[i].shape)

			print("-"*20)
		
		# ----------------------------------------------------		
		# APPLY THE MULTI CHANNELS ABSTRACTION (DECONVOLUTION)
		# ----------------------------------------------------
		if config["TG"]:
			merged = concatenate([sent_representation[0], sent_representation[1], sent_representation[2]])
			#merged = multiply([conv[0], conv[1], conv[2]])
			print("merged", merged.shape)
		else:
			merged = sent_representation[0]

		# ----------
		# LSTM LAYER
		# ----------
		"""
		lstm = LSTM(config["LSTM_SIZE"], return_sequences=True)(merged)
		print("lstm :", lstm.shape)

		# ---------------
		# ATTENTION LAYER
		# ---------------
		attention = TimeDistributed(Dense(1, activation='tanh'))(lstm) 
		print("TimeDistributed :", attention.shape)

		# reshape Attention
		attention = Flatten()(attention)
		print("Flatten :", attention.shape)
		
		attention = Activation('softmax')(attention)
		print("Activation :", attention.shape)

		# Observe attention here
		#attention_model = Model(inputs=inputs, outputs=attention)

		# Pour pouvoir faire la multiplication (scalair/vecteur KERAS)
		attention = RepeatVector(config["LSTM_SIZE"])(attention)
		print("RepeatVector :", attention.shape)
		
		attention = Permute([2, 1])(attention)
		print("Permute :", attention.shape)

		# apply the attention		
		sent_representation = multiply([lstm, attention])
		print("Multiply :", sent_representation.shape)
		
		sent_representation = Lambda(lambda xin: K.sum(xin, axis=2))(sent_representation)
		print("Lambda :", sent_representation.shape)
		"""
		# -------------
		# DROPOUT LAYER
		# -------------
		#if config["ENABLE_LSTM"]:
		#	dropout = Dropout(config["DROPOUT_VAL"])(sent_representation)
		#else:
		dropout = Flatten()(merged)
		dropout = Dropout(config["DROPOUT_VAL"])(dropout)
		print("Dropout :", dropout.shape)

		# -----------------
		# HIDDEN DENSE LAYER
		# -----------------		
		hidden_dense = Dense(config["DENSE_LAYER_SIZE"], kernel_initializer='uniform',activation='relu')(dropout)

		# -----------------
		# FINAL DENSE LAYER
		# -----------------
		crossentropy = 'categorical_crossentropy'
		output_acivation = 'softmax'

		output = Dense(config["num_classes"], activation=output_acivation)(hidden_dense)
		print("output :", output.shape)

		# -----------------
		# COMPILE THE MODEL
		# -----------------
		model = Model(inputs=inputs, outputs=output)
		op = optimizers.Adam(lr=config["LEARNING_RATE"], beta_1=0.9, beta_2=0.999, epsilon=1e-08)
		model.compile(optimizer=op, loss=crossentropy, metrics=['accuracy'])

		print("-"*20)
		print("MODEL READY")
		print("-"*20)

		print("TRAINING MODEL")
		print(model.summary())

		return model