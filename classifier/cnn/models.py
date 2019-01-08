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
from keras.layers import Conv1D, Conv2D, Conv2DTranspose, MaxPooling1D, MaxPooling2D, GlobalMaxPooling1D,  Embedding, Reshape
from keras.layers import Input, Embedding, LSTM, Dense
from keras.utils import np_utils
from keras.layers import multiply

#from config import EMBEDDING_DIM, config["NB_FILTERS"], config["FILTER_SIZES"], DROPOUT_VAL

class CNNModel:
	
	def getModel(self, config, weight=None):

		print("-"*20)
		print("CREATE MODEL")
		print("-"*20)
		
		inputs = Input(shape=(config["SEQUENCE_SIZE"],), dtype='int32')
		
		# ---------------
		# EMBEDDING LAYER
		# ---------------
		embedding = Embedding(
			config["vocab_size"]+1, # due to mask_zero
			config["EMBEDDING_DIM"],
			input_length=config["SEQUENCE_SIZE"],
			weights=weight,
			trainable=True
		)(inputs)
		print("embedding : ", embedding.shape)
		
		reshape = Reshape((config["SEQUENCE_SIZE"],config["EMBEDDING_DIM"],1))(embedding)
		print("reshape : ", reshape.shape)

		# ---------------------------------------
		# CONVOLUTION (FOR CNN+LSTM MODEL)
		# ---------------------------------------
		filter = config["FILTER_SIZES"]
		if config["TG"]:
			conv_width = int(config["EMBEDDING_DIM"]/3)
		else:
			conv_width = config["EMBEDDING_DIM"]
		conv = Conv2D(config["NB_FILTERS"], (filter, conv_width), strides=(1, conv_width), padding='valid', kernel_initializer='normal', activation='relu', data_format='channels_last')(reshape)	
		print("convolution :", conv.shape)
		
		# -----------------------------------------
		# DECONVOLUTION (FOR CNN+LSTM MODEL)
		# -----------------------------------------
		deconv = Conv2DTranspose(1, (filter, conv_width), strides=(1, conv_width), padding='valid', kernel_initializer='normal', activation='relu', data_format='channels_last')(conv)
		print("deconvolution :", deconv.shape)
		deconv_model = Model(inputs=inputs, outputs=deconv)

		# --------------------
		# ! ONLY FOR CNN MODEL
		# --------------------	
		maxpool = MaxPooling2D(pool_size=(config["SEQUENCE_SIZE"] - filter + 1, 1), strides=(1, conv_width), padding='valid', data_format='channels_last')(conv)
		print("MaxPooling2D :", maxpool.shape)
		maxpool = Flatten()(maxpool)
		print("flatten :", maxpool.shape)

		# ---------------------
		# ! ONLY FOR LSTM MODEL
		# ---------------------		
		conv_shape = conv.shape[1:]
		reshape = Reshape((int(conv_shape[0]),int(np.prod(conv_shape[1:]))))(conv)
		print("reshape :", reshape.shape)

		# ----------
		# LSTM LAYER
		# ----------
		# Select input as "reshape" to use CONVOLUTION
		# Select input as "embedding" to drop CONVOLUTION
		if (config["ENABLE_CONV"]):
			lstm = LSTM(config["LSTM_SIZE"], return_sequences=True)(reshape) #(embedding) <=== Select here with or without convolution
		else:
			lstm = LSTM(config["LSTM_SIZE"], return_sequences=True)(embedding)
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
		attention_model = Model(inputs=inputs, outputs=attention)

		# Pour pouvoir faire la multiplication (scalair/vecteur KERAS)
		attention = RepeatVector(config["LSTM_SIZE"])(attention)
		print("RepeatVector :", attention.shape)
		
		attention = Permute([2, 1])(attention)
		print("Permute :", attention.shape)

		# apply the attention		
		sent_representation = multiply([lstm, attention])
		print("Multiply :", sent_representation.shape)
		
		sent_representation = Lambda(lambda xin: K.sum(xin, axis=1))(sent_representation)
		print("Lambda :", sent_representation.shape)

		# -------------
		# DROPOUT LAYER
		# -------------
		# Select input as "sent_representation" to use LSTM
		# Select input as "maxpool" to drop LSTM
		if config["ENABLE_LSTM"]:
			dropout = Dropout(config["DROPOUT_VAL"])(sent_representation)
		else:
			dropout = Dropout(config["DROPOUT_VAL"])(maxpool)
		print("Dropout :", dropout.shape)


		# -----------------
		# HIDDEN DENSE LAYER
		# -----------------		
		hidden_dense = Dense(config["DENSE_LAYER_SIZE"],kernel_initializer='uniform',activation='relu')(dropout)

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

		print("DECONV MODEL")
		print(deconv_model.summary())

		print("ATTENTION MODEL")
		print(attention_model.summary())

		return model, deconv_model, attention_model
	