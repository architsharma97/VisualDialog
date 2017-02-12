'''
Code modelled along implementation of LSTM layer by Ryan Kiros
(https://github.com/ryankiros/)
'''

import sys
sys.path.append('../')

import numpy as np
import theano.tensor as T

from utils import init_weights, _concat

def param_init_fflayer(params, prefix, nin, nout):
	'''
	Initializes weights for a feedforward layer
	'''
	params[_concat(prefix,'W')]=init_weights(nin, nout, type_init='ortho')
	params[_concat(prefix,'b')]=np.zeros((nout,)).astype('float32')

	return params

def fflayer(params, state_below, prefix):
	'''
	A feedforward layer with tanh nonlinearity
	'''
	return T.tanh(T.dot(state_below, params[_concat(prefix, 'W')])+params[_concat(prefix, 'b')])

def param_init_lstm(params, prefix, nin, units):
	'''
	Weight initialization for a canonical LSTM layer
	'''
	# concatenate the weight for all gates into one matrix for faster training
	W = np.concatenate([init_weights(nin, units),
		init_weights(nin, units),
		init_weights(nin, units),
		init_weights(nin, units)],
		axis=1)
	
	params[_concat(prefix, 'W')] = W

	U = np.concatenate([init_weights(nin, units, type_init='ortho'),
		init_weights(nin, units, type_init='ortho'),
		init_weights(nin, units, type_init='ortho'),
		init_weights(nin, units, type_init='ortho')],
		axis=1)
	params[_concat(prefix, 'U')] = U
	params[_concat(prefix, 'b')] = np.zeros((4*units,)).astype('float32')
	
	return params