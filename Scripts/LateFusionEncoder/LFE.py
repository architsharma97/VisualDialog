ADD print statments otherwise this script will not run
import sys
sys.path.append('../')

import numpy as np
import theano.tensor as tensor

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

# load the embeddings
embeddings=np.load(DATA_DIR+'embedding_matrix.npy').astype('float32')
EMBEDDINGS_DIM=embeddings.shape[1]

# loading dictionaries
word_idx_map=load_obj(DATA_DIR+'dictionary.pkl')
idx_word_map=load_obj(DATA_DIR+'reverse_dictionary.pkl')

# preprocess the training data to get input matrices and tensors
image_features, questions_tensor, answers_tensor=preprocess(DATA_DIR, load_dict=True, load_embedding_data=True, save_data=False)

# Parameters for the model
params=OrderedDict()

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
	tparams[key]=theano.shared(params[key], name=key)

# Construction of theano computation graph using symbolic variables