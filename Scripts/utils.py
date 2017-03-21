import cPickle as pickle
import numpy as np
import theano.tensor as T

# general function to save python objects using pickle
def save_obj(obj, address):
	with open(address, 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# general function to load python objects using pickle
def load_obj(address):
	with open(address, 'rb')as f:
		return pickle.load(f)

def init_weights(nin, nout, type_init='uni', scale=0.1):
	'''
	type_init={'uni' : uniform initialization between [-scale,scale],
		  'ortho': orthogonal weight initialization, initializes uniformly if nin!=nout }
	'''
	if nin == nout and type_init == 'ortho':
		W = np.random.randn(nin, nin)
		W, s, v = np.linalg.svd(W)
	else:
		W = np.random.uniform(low=-scale, high=scale, size=(nin, nout))

	return W.astype('float32')

def _concat(str1, str2):
	'''
	Returns str1_str2
	'''
	return '%s_%s' % (str1,str2)