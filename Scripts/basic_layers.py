'''
Code modelled along implementation of LSTM layer by Ryan Kiros
(https://github.com/ryankiros/)
'''

import sys
sys.path.append('../')

import numpy as np
import theano
import theano.tensor as T

from utils import init_weights, _concat

def param_init_fflayer(params, prefix, nin, nout):
	'''
	Initializes weights for a feedforward layer
	'''
	params[_concat(prefix,'W')] = init_weights(nin, nout, type_init='ortho')
	params[_concat(prefix,'b')] = np.zeros((nout,)).astype('float32')

	return params

def fflayer(tparams, state_below, prefix):
	'''
	A feedforward layer with tanh nonlinearity
	'''
	return T.tanh(T.dot(state_below, tparams[_concat(prefix, 'W')]) + tparams[_concat(prefix, 'b')])

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

	U = np.concatenate([init_weights(units, units, type_init='ortho'),
		init_weights(units, units, type_init='ortho'),
		init_weights(units, units, type_init='ortho'),
		init_weights(units, units, type_init='ortho')],
		axis=1)
	
	params[_concat(prefix, 'U')] = U
	
	params[_concat(prefix, 'b')] = np.zeros((4 * units,), dtype=np.float32)
	
	return params

def lstm_layer(tparams, state_below, 
			   prefix,
			   mask=None,
			   init_state=None, 
			   init_memory=None,
			   one_step=False, 
			   n_steps=None):
	'''
	Defines the forward pass of a LSTM for a sequence of questions/history (after passing through the embedding)
	state_below: timesteps x samples x embedding_size
	Use a mask if input contains sequences of variable length.
	'''
	if n_steps is None:
		n_steps = state_below.shape[0]

	if state_below.ndim == 3:
		n_samples = state_below.shape[1]
	else:
		n_samples = 1

	dim = tparams[_concat(prefix,'U')].shape[0]
	
	if init_state is None:
		init_state = T.alloc(0., n_samples, dim)

	if init_memory is None:
		init_memory = T.alloc(0., n_samples, dim)

	U = tparams[_concat(prefix, 'U')]
	b = tparams[_concat(prefix, 'b')]
	W = tparams[_concat(prefix, 'W')]
	
	non_seq = [U, b, W]

	def _slice(_x, n, dim):
		if _x.ndim == 3:
			return _x[:, :, n*dim:(n+1)*dim]
		return _x[:, n*dim:(n+1)*dim]

	def _step(mask, sbelow, sbefore, cell_before, *args):
		preact = T.dot(sbefore, U)
		preact += sbelow
		preact += b

		i = T.nnet.sigmoid(_slice(preact, 0, dim))
		f = T.nnet.sigmoid(_slice(preact, 1, dim))
		o = T.nnet.sigmoid(_slice(preact, 2, dim))
		c = T.tanh(_slice(preact, 3, dim))

		# should arise in decoding situations only
		if mask is None:
			c = f * cell_before + i * c
			h = o * T.tanh(c)
		else:
			c = f * cell_before + i * c
			c = (mask * (c.T) + (1. - mask) * (cell_before.T)).T
			h = o * T.tanh(c)
			h = (mask * (h.T) + (1. - mask) * (sbefore.T)).T
			
		return h, c

	lstm_state_below = T.dot(state_below, W) + b
	if state_below.ndim == 3:
		lstm_state_below = lstm_state_below.reshape((state_below.shape[0], 
													 state_below.shape[1], 
													 -1))
	# mainly for decoder
	if one_step:
		h, c = _step(mask, lstm_state_below, init_state, init_memory)
		return h, c
	
	if mask is None:
		mask = T.alloc(1., n_steps, n_samples)

	outs, updates = theano.scan(_step, 
								sequences=[mask, lstm_state_below],
								outputs_info=[init_state, init_memory],
								name=_concat(prefix, 'layers'),
								non_sequences=non_seq,
								strict=True,
								n_steps=n_steps)
	
	return outs