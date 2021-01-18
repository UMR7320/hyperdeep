import numpy as np

from keras import optimizers
from keras import regularizers

from keras import backend as K
from keras.models import Sequential,Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import Permute
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Conv1D, UpSampling1D, Conv2D, Conv2DTranspose, MaxPooling1D, MaxPooling2D, GlobalMaxPooling1D,  Embedding, Reshape, Dropout
from tensorflow.keras.layers import Input, Embedding, LSTM, GRU, Bidirectional, Dense
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import concatenate
from keras.utils import np_utils
from tensorflow.keras.layers import multiply
from tensorflow.keras.layers import BatchNormalization
from keras.utils import multi_gpu_model

from tensorflow.python.client import device_lib

class Language:
	
	def getModel(self, config, input_size, output_size):

		print("-"*20)
		print("CREATE MODEL")
		print("-"*20)
		
		# INPUTS
		inputs = Input(shape=(input_size, output_size, ), dtype='float32')
		print("input",  inputs.shape)

		# ----------
		# LSTM LAYER
		# ----------
		rnn1 =LSTM(config["LSTM_SIZE"], return_sequences=True, dropout=config["DROPOUT_VAL"], recurrent_dropout=config["DROPOUT_VAL"])(inputs)


		rnn2 =LSTM(config["LSTM_SIZE"], return_sequences=False, dropout=config["DROPOUT_VAL"], recurrent_dropout=config["DROPOUT_VAL"])(rnn1)

		# ---------------
		# ATTENTION LAYER
		# ---------------
		"""
		attention = TimeDistributed(Dense(1, activation='tanh'))(rnn) 
		print("TimeDistributed :", attention.shape)

		# Apply Attention
		attention = Flatten()(attention)
		attention = Activation('softmax')(attention)
		attention = RepeatVector(config["LSTM_SIZE"]*2)(attention)
		attention = Permute([2, 1])(attention)	
		sent_representation = multiply([rnn, attention])
		
		flat = Flatten()(sent_representation)
		"""
	
		# -------------
		# DROPOUT LAYER
		# -------------
		#dropout = Dropout(config["DROPOUT_VAL"])(flat)

		# -----------------
		# FINAL DENSE LAYER
		# -----------------
		output = Dense(output_size, activation='softmax')(rnn2) #, kernel_regularizer=regularizers.l1(0.05)

		print("output :", output.shape)

		# -----------------
		# COMPILE THE MODEL
		# -----------------
		model = Model(inputs=inputs, outputs=output)		

		crossentropy = 'categorical_crossentropy'
		op = optimizers.Adam(lr=config["LEARNING_RATE"], beta_1=0.9, beta_2=0.999, epsilon=1e-08)
		model.compile(optimizer=op, loss=crossentropy, metrics=['accuracy'])

		print("-"*20)
		print("MODEL READY")
		print("-"*20)

		print("TRAINING MODEL")
		print(model.summary())

		return model