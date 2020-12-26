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
from tensorflow.keras.layers import Conv1D, UpSampling1D, Conv2D, Conv2DTranspose, MaxPooling1D, MaxPooling2D, GlobalMaxPooling1D,  Embedding, Reshape
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
		rnn = GRU(config["LSTM_SIZE"])(inputs)
		print("rnn :", rnn.shape)

		# -----------------
		# FINAL DENSE LAYER
		# -----------------
		output = Dense(output_size, activation='softmax')(rnn) #, kernel_regularizer=regularizers.l1(0.05)

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