ADD print statments otherwise this script will not run
import sys
sys.path.append('../')

import numpy as np
import theano.tensor as T

from utils import init_weights, _concat, load_obj
from preprocess import preprocess
from basic_layers import param_init_fflayer, param_init_lstm, fflayer, lstm_layer

from collections import OrderedDict

# constants
DATA_DIR='../../Data/'

# VGG16 Specification
IMAGE_DIM=4096

# LSTMs Specifications: H=>QA-History+Caption Encoder, Q=>Question Encoder 
LSTM_H_OUT=512
LSTM_H_LAYERS=2
lstm_prefix_h='lstm_h'

LSTM_Q_OUT=512
LSTM_Q_LAYERS=2
lstm_prefix_q='lstm_q'

# Fully Connected Layer Specification
FF_IN=IMAGE_DIM+LSTM_Q_OUT+LSTM_H_OUT
FF_OUT=512
ff_prefix='ff'

def initialize_lfe():
	'''
	Initialize the parameters for the late fusion encoder
	'''
	# Parameters for the model
	params=OrderedDict()
	global (DATA_DIR, IMAGE_DIM, LSTM_H_OUT, LSTM_H_LAYERS, lstm_prefix_h, 
		lstm_prefix_q, LSTM_Q_LAYERS, LSTM_Q_OUT, FF_IN, 
		FF_OUT, ff_prefix)
	
	# feedforward layer
	params=param_init_fflayer(params, ff_prefix, FF_IN, FF_OUT)

	# lstm layers
	# question encoding layers
	params=param_init_lstm(params, _concat(lstm_prefix_q, 1), EMBEDDINGS_DIM, LSTM_Q_OUT)
	params=param_init_lstm(params, _concat(lstm_prefix_q, 2), LSTM_Q_OUT, LSTM_Q_OUT)

	# history encoding layers
	params=param_init_lstm(params, _concat(lstm_prefix_h, 1), EMBEDDINGS_DIM, LSTM_H_OUT)
	params=param_init_lstm(params, _concat(lstm_prefix_h, 2), LSTM_H_OUT, LSTM_H_OUT)

	# Initialize Theano Shared Variables for params
	tparams=OrderedDict()
	for key, val in params.iteritems():
		tparams[key]=T.shared(params[key], name=key)

	return tparams

def build_lfe(tparams):
	'''
	Builds the computational graph of the encoder
	'''
	global (DATA_DIR, IMAGE_DIM, LSTM_H_OUT, LSTM_H_LAYERS, lstm_prefix_h, 
		lstm_prefix_q, LSTM_Q_LAYERS, LSTM_Q_OUT, FF_IN, 
		FF_OUT, ff_prefix)

	# vgg16 extracted features for images
	# data preparation ensures that the number of images matches the number of questions
	img = T.matrix('img', dtype='float32')
	# steps x samples x dimensions
	que = T.dtensor3('que', dtype='float32')
	his = T.dtensor3('his', dtype='float32')

	qsteps = que.shape[0]
	hsteps = his.shape[0]

	# encoding questions
	out_1 = lstm_layer(tparams, que, _concat(lstm_prefix_q, 1), n_steps=qsteps)
	
	# restructure
	in_2 = T.zeros((qsteps, que.shape[1], out_1[0][0].shape[1]), dtype='float32')
	for i in range(len(out_1)):
		in_2[i,:,:] = out_1[i][0]

	out_2 = lstm_layer(tparams, in_2, _concat(lstm_prefix_q, 2), n_steps=qsteps)

	# samples x dim_projection
	qcode = out_2[-1][0]

	# encoding history
	out_3 = lstm_layer(tparams, his, _concat(lstm_prefix_h, 1), n_steps=hsteps)
	
	# restructure
	in_4 = T.zeros((hsteps, his.shape[1], out_3[0][0].shape[1]), dtype='float32')
	for i in range(len(out_3)):
		in_4[i,:,:] = out_3[i][0]

	out_4 = lstm_layer(tparams, in_4, _concat(lstm_prefix_h, 2), n_steps=hsteps)

	# samples x dim_projection
	hcode = out_4[-1][0]

	# late fusion: concat of hcode, qcode and img
	in_5 = T.concatenate([img, qcode, hcode], axis=1)
	lfcode = fflayer(tparams, in_5, ff_prefix)

	return img, que, his, lfcode


# load the embeddings
embeddings=np.load(DATA_DIR+'embedding_matrix.npy').astype('float32')
EMBEDDINGS_DIM=embeddings.shape[1]

# loading dictionaries
word_idx_map=load_obj(DATA_DIR+'dictionary.pkl')
idx_word_map=load_obj(DATA_DIR+'reverse_dictionary.pkl')

# preprocess the training data to get input matrices and tensors
image_features, questions_tensor, answers_tensor=preprocess(DATA_DIR, load_dict=True, load_embedding_data=True, save_data=False)
tparams=initialize()