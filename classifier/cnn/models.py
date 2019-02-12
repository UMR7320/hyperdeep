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
from keras.layers import Lambda
from keras.layers import concatenate
from keras.utils import np_utils
from keras.layers import multiply

class CNNModel:
	
	def getModel(self, config, weight=None):

		print("-"*20)
		print("CREATE MODEL")
		print("-"*20)

		"""
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
		
		reshape = Reshape((config["SEQUENCE_SIZE"], config["EMBEDDING_DIM"], 1))(embedding)
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
		"""
		
		# ---------------------------------------
		# MULTI CHANNELS CONVOLUTION
		# ---------------------------------------
		inputs = [0]*3
		embedding = [0]*3
		reshape = [0]*3
		conv = [0]*3
		deconv = [0]*3
		deconv_model = [0]*3
		pool = [0]*3
		flat = [0]*3

		for i, arg in enumerate(["F", "C", "L"]):
			print("CHANNELS ", i)
			inputs[i] = Input(shape=(config["SEQUENCE_SIZE"],), dtype='int32')
			print("input", i,  inputs[i].shape)
			embedding[i] = Embedding(
				config["vocab_size"][i],
				config[arg + "_EMBEDDING_DIM"],
				input_length=config["SEQUENCE_SIZE"],
				#weights=weight[i],
				trainable=True
			)(inputs[i])
			print("embedding", i,  embedding[i].shape)
			reshape[i] = Reshape((config["SEQUENCE_SIZE"], config[arg + "_EMBEDDING_DIM"], 1))(embedding[i])
			print("reshape", i,  reshape[i].shape)
			conv[i] = Conv2D(config[arg + "_NB_FILTERS"], (config[arg + "_FILTER_SIZES"], config[arg + "_EMBEDDING_DIM"]), padding='valid', kernel_initializer='normal', activation='relu', data_format='channels_last')(reshape[i])
			print("conv", i,  conv[i].shape)
			deconv[i] = Conv2DTranspose(1, (config[arg + "_FILTER_SIZES"], config[arg + "_EMBEDDING_DIM"]), padding='valid', kernel_initializer='normal', activation='relu', data_format='channels_last')(conv[i])
			deconv_model[i] = Model(inputs=inputs[i], outputs=deconv[i])
			pool[i] = MaxPooling2D(pool_size=(config["SEQUENCE_SIZE"] - config[arg + "_FILTER_SIZES"] + 1, 1), strides=(1, config[arg + "_EMBEDDING_DIM"]), padding='valid', data_format='channels_last')(conv[i])
			print("pool", i,  pool[i].shape)
			flat[i] = Flatten()(pool[i])
			print("flat", i,  flat[i].shape)
			print("-"*20)
		
		# merge
		merged = concatenate([flat[0], flat[1], flat[2]])
		print("merged", merged.shape)

		# dropout
		dropout = Dropout(config["DROPOUT_VAL"])(merged)
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

		#print("DECONV MODEL")
		#print(deconv_model.summary())

		#print("ATTENTION MODEL")
		#print(attention_model.summary())

		return model, deconv_model