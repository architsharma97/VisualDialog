import sys
sys.path.append('../')

import numpy as np
import theano.tensor as T
import theano

from utils import init_weights, _concat, load_obj
from preprocess import preprocess
from basic_layers import param_init_fflayer, param_init_lstm, fflayer, lstm_layer
from adam import adam

from collections import OrderedDict

print "Initializing constants"
# constants
DATA_DIR = '../../Data/'
MODEL_DIR = '../../Models/'

# VGG16 Specification
IMAGE_DIM = 4096

# LSTMs Specifications: H=>QA-History+Caption Encoder, Q=>Question Encoder 
LSTM_H_OUT = 512
LSTM_H_LAYERS = 2
lstm_prefix_h = 'lstm_h'

LSTM_Q_OUT = 512
LSTM_Q_LAYERS = 2
lstm_prefix_q = 'lstm_q'

# fully Connected Layer Specification
FF_IN = IMAGE_DIM + LSTM_Q_OUT + LSTM_H_OUT
FF_OUT = 512
ff_prefix = 'ff'

# decoder layers
LSTM_D = FF_OUT
LSTM_D_LAYERS = 2
lstm_prefix_d = 'lstm_d'
MAX_TOKENS = 60

# maximum gradient allowed in a step
GRAD_CLIP = 5.0

print "Loading embedding matrix"
try:
	embeddings = T.as_tensor_variable(np.transpose(np.load(MODEL_DIR + 'embedding_matrix.npy').astype('float32')))
	load_embedding_data = True
except:
	print "Unable to load embedding matrix\nWill be created after preprocessing"
	load_embedding_data = False

print "Loading dictionaries"
try:
	# loading dictionaries
	word_idx_map = load_obj(MODEL_DIR + 'dictionary.pkl')
	idx_word_map = load_obj(MODEL_DIR + 'reverse_dictionary.pkl')
	load_dict = True
except:
	print "Unable to load dictionaries\nWill be loaded after preprocessing"
	load_dict = False

# preprocess the training data to get input matrices and tensors
image_features, questions_tensor, answers_tokens_idx = preprocess(DATA_DIR, load_dict=load_dict, load_embedding_matrix=load_embedding_data, save_data=False)

if not load_embedding_data:
	print "Loading embedding matrix"
	embeddings = T.as_tensor_variable(np.transpose(np.load(MODEL_DIR + 'embedding_matrix.npy').astype('float32')))

if not load_dict:
	print "Loading dictionaries"
	word_idx_map = load_obj(MODEL_DIR + 'dictionary.pkl')
	idx_word_map = load_obj(MODEL_DIR + 'reverse_dictionary.pkl')

EMBEDDINGS_DIM = embeddings.shape[0]

def initialize():
	'''
	Initialize the parameters for the late fusion encoder and decoder
	'''
	# Parameters for the model
	params = OrderedDict()
	global DATA_DIR, IMAGE_DIM, LSTM_H_OUT, LSTM_H_LAYERS, lstm_prefix_h, \
		lstm_prefix_q, LSTM_Q_LAYERS, LSTM_Q_OUT, FF_IN, \
		LSTM_D, LSTM_D_LAYERS, lstm_prefix_d, EMBEDDINGS_DIM, FF_OUT, ff_prefix
	
	# feedforward layer
	params = param_init_fflayer(params, ff_prefix, FF_IN, FF_OUT)

	# lstm layers
	# question encoding layers
	params = param_init_lstm(params, _concat(lstm_prefix_q, 1), EMBEDDINGS_DIM, LSTM_Q_OUT)
	params = param_init_lstm(params, _concat(lstm_prefix_q, 2), LSTM_Q_OUT, LSTM_Q_OUT)

	# history encoding layers
	params = param_init_lstm(params, _concat(lstm_prefix_h, 1), EMBEDDINGS_DIM, LSTM_H_OUT)
	params = param_init_lstm(params, _concat(lstm_prefix_h, 2), LSTM_H_OUT, LSTM_H_OUT)

	# decoding layers
	params = param_init_lstm(params, _concat(lstm_prefix_d, 1), EMBEDDINGS_DIM, LSTM_D)
	params = param_init_lstm(params, _concat(lstm_prefix_d, 2), LSTM_D, EMBEDDINGS_DIM)

	# initialize theano shared variables for params
	tparams = OrderedDict()
	for key, val in params.iteritems():
		tparams[key] = T.shared(params[key], name=key)

	return tparams

def build_lfe(tparams):
	'''
	Builds the computational graph of the encoder
	'''
	
	global DATA_DIR, IMAGE_DIM, LSTM_H_OUT, LSTM_H_LAYERS, lstm_prefix_h, \
		lstm_prefix_q, LSTM_Q_LAYERS, LSTM_Q_OUT, FF_IN, \
		LSTM_D, LSTM_D_LAYERS, lstm_prefix_d, EMBEDDINGS_DIM, FF_OUT, ff_prefix

	# data preparation ensures that the number of images matches the number of questions
	img = T.matrix('img', dtype='float32')
	
	# steps x samples x dimensions
	que = T.tensor3('que', dtype='float32')
	his = T.tensor3('his', dtype='float32')

	qsteps = que.shape[0]
	hsteps = his.shape[0]

	# encoding questions
	out_1 = lstm_layer(tparams, que, _concat(lstm_prefix_q, 1), n_steps=qsteps)
	
	# restructure
	in_2 = T.as_tensor_variable([array[0] for array in out_1])

	out_2 = lstm_layer(tparams, in_2, _concat(lstm_prefix_q, 2), n_steps=qsteps)

	# samples x dim_projection
	qcode = out_2[-1][0]

	# encoding history
	out_3 = lstm_layer(tparams, his, _concat(lstm_prefix_h, 1), n_steps=hsteps)
	
	# restructure
	in_4 = T.as_tensor_variable([array[0] for array in out_3])

	out_4 = lstm_layer(tparams, in_4, _concat(lstm_prefix_h, 2), n_steps=hsteps)

	# samples x dim_projection
	hcode = out_4[-1][0]

	# late fusion: concat of hcode, qcode and img2
	in_5 = T.concatenate([img, qcode, hcode], axis=1)
	lfcode = fflayer(tparams, in_5, ff_prefix)

	return img, que, his, lfcode

def build_decoder(tparams, lfcode, max_steps):
	'''
	Builds computational graph for generative decoder
	'''
	
	global DATA_DIR, IMAGE_DIM, LSTM_H_OUT, LSTM_H_LAYERS, lstm_prefix_h, \
		lstm_prefix_q, LSTM_Q_LAYERS, LSTM_Q_OUT, FF_IN, embeddings, \
		LSTM_D, LSTM_D_LAYERS, lstm_prefix_d, EMBEDDINGS_DIM, FF_OUT, ff_prefix, MAX_TOKENS, word_idx_map

	def _decode_step(sbelow, sbefore_1, sbefore_2, cell_before_1, cell_before_2):
		'''
		Custom step for decoder, where the output of 2nd LSTM layer is fed into the 1st LSTM layer
		'''
		out_1 = lstm_layer(tparams, sbelow, _concat(lstm_prefix_d, 1), init_state=sbefore_1, init_memory=cell_before_1, one_step=True)

		out_2 = lstm_layer(tparams, out_1[0][0], _concat(lstm_prefix_d, 1), init_state=sbefore_2, init_memory=cell_before_2, one_step=True)

		return out_2[0][0], out_1[0][0], out_2[0][0], out_1[0][1], out_2[0][1]

	def softmax(inp):
		'''
		Chooses the right element from the outputs for softmax
		'''
		return T.nnet.softmax(T.dot(inp[2], embeddings))

	n_samples = lfcode.shape[0]
	hdim = lfcode.shape[1]

	memory_1 = T.as_tensor_variable(np.zeros((n_samples, hdim)).astype('float32'))
	memory_2 = T.as_tensor_variable(np.zeros((n_samples, hdim)).astype('float32'))

	dim = embeddings.shape[1]

	init_token = T.as_tensor_variable(np.tile(embeddings[word_idx_map['<sos>']], (n_samples, 1)))

	# initial hidden state for both 1st and 2nd layer is lfcode
	tokens, updates = theano.scan(_decode_step,
									output_info=[init_token, lfcode, lfcode, memory_1, memory_2],
									strict=True,
									n_steps=max_steps)

	soft_tokens, updates = theano.scan(softmax, sequences=tokens)

	return T.as_tensor_variable(soft_tokens)

print "Initializating parameters for model"
tparams = initialize()

print "Building encoder for the model"
img, que, his, lfcode = build_lfe(tparams)

# answer tensor should be a binary tensor with 1's at the positions which needs to be included
# timesteps x number of answers in minibatch x vocabulary size 
ans = T.tensor3('ans', dtype='float32')

print "Building decoder"
pred = build_decoder(tparams, lfcode, MAX_TOKENS)

# cost function
cost = -T.log(pred*ans).sum()
inps = [img, que, his, pred]

print "Constructing graph"
f_cost = theano.function(inps, cost, profile=False)
print "Done!"

print "Computing gradients"
param_list=[val for key, val in tparams.iteritems()]
grads = T.grad(cost, wrt=param_list)

# computinng norms
f_grad_norm = theano.function(inps, [(g**2).sum() for g in grads], profile=False)
f_weight_norm = theano.function([], [(v**2).sum() for k,v in tparams.iteritems()], profile=False)

# gradient is clipped beyond certain values
g2 = 0.
for g in grads:
	g2 += (g**2).sum()
new_grads = []
for g in grads:
	new_grads.append(T.switch(g2 > (GRAD_CLIP**2),
								g / T.sqrt(g2)*GRAD_CLIP, g))
grads = new_grads

# learning rate
lr = T.scale(name='lr', dtype='float32')

# gradients, update parameters
print "Setting up optimizer"
f_grad_shared, f_update = adam(lr, tparams, grads, inps, cost)