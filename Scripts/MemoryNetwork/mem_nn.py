import sys
sys.path.append('../')

import numpy as np
import theano.tensor as T
import theano

from utils import init_weights, _concat, load_obj
from preprocess import preprocess
from basic_layers import param_init_fflayer, param_init_lstm, fflayer, lstm_layer
from adam import adam
import minibatch_memNN

from collections import OrderedDict
import time

'''
Script takes the following arguments if being used for validation
No arguments should be passed if the script is used for training from random initialization
1) Train (0) or Validate (1)
2) Location of the weights (.npz)
3) Location where the ranks for the correct answer are stored for all validation data (optional, recommended) (not to be given with training)
'''

if len(sys.argv) > 1 and int(sys.argv[1]) == 1:
	print "Running in validation mode"
	val = True
else:
	print "Running in train mode"
	val = False

# initializing constants
DATA_DIR = '../../Data/'
MODEL_DIR = '../../Models/'

# VGG16 Specification
IMAGE_DIM = 4096

# LSTMs Specifications: H := QA-History+Captions Encoder, Q := Question Encoder 
LSTM_H_OUT = 512
LSTM_H_LAYERS = 2
lstm_prefix_h = 'lstm_h'

LSTM_Q_OUT = 512
LSTM_Q_LAYERS = 2
lstm_prefix_q = 'lstm_q'

# fully Connected Layer Specification
FF_IN = IMAGE_DIM + LSTM_Q_OUT
FF_OUT = 512
ff_prefix = 'ff'

# decoder layers
LSTM_D_OUT = 512
LSTM_D_LAYERS = 2
lstm_prefix_d = 'lstm_d'
MAX_TOKENS = 60

# maximum gradient allowed in a step
GRAD_CLIP = 5.0

# number of epochs
EPOCHS = 150

# training parameters
reduced_instances = 1
learning_rate = 0.001

print "Loading embedding matrix"
try:
	embed = np.transpose(np.load(MODEL_DIR + 'embedding_matrix.npy').astype('float32'))
	embeddings = T.as_tensor_variable(embed)
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

print "Preprocessing data"
if len(sys.argv) <=1 or int(sys.argv[1]) == 0:
	# preprocess the training data to get input matrices and tensors
	image_features, questions_tensor, answers_tensor, answers_tokens_idx = preprocess(DATA_DIR,
																		   load_dict=load_dict,
																		   load_embedding_matrix=load_embedding_data,
																		   save_data=False,
																		   reduced_instances=reduced_instances)
	print 'Shape of image features: ', image_features.shape
	print 'Shape of questions_tensor: ', questions_tensor.shape
	print 'Shape of answers_tensor: ', answers_tensor.shape
	print 'Shape of answers_tokens_idx: ', answers_tokens_idx.shape
else:
	image_features, captions, questions, answers_options, correct_options = preprocess(DATA_DIR,
																		 load_dict=True,
																		 load_embedding_matrix=True,
																		 split='Val',
																		 save_data=False,
																		 reduced_instances=-1)
	print 'Number of images: ', image_features.shape[0]

if not load_embedding_data:
	print "Loading embedding matrix"
	embed = np.transpose(np.load(MODEL_DIR + 'embedding_matrix.npy').astype('float32'))
	embeddings = T.as_tensor_variable(embed)

if not load_dict:
	print "Loading dictionaries"
	word_idx_map = load_obj(MODEL_DIR + 'dictionary.pkl')
	idx_word_map = load_obj(MODEL_DIR + 'reverse_dictionary.pkl')

EMBEDDINGS_DIM = embed.shape[0]

if len(sys.argv) <=1 or int(sys.argv[1]) == 0:
	print "Preparing minibatches"
	train_data = minibatch_memNN.data(image_features, questions_tensor, answers_tensor, answers_tokens_idx, len(idx_word_map), batch_size=64)

def initialize(address=None):
	'''
	Initilize parameters for the memory network models
	address: location of parameters to be loaded. Default is to reinitialize.
	'''

	if address is None:
		params = OrderedDict()

		global ff_prefix, FF_OUT, FF_IN, lstm_prefix_q, EMBEDDINGS_DIM, LSTM_Q_OUT, \
					lstm_prefix_h, LSTM_H_OUT, lstm_prefix_d, LSTM_D_OUT
		
		# feedforward layer for question and image to generate the query vector
		params = param_init_fflayer(params, _concat(ff_prefix, 1), FF_IN, FF_OUT)
		# feedforward layer for memory vector
		params = param_init_fflayer(params, _concat(ff_prefix, 2), LSTM_H_OUT, FF_OUT)
		
		# lstm layers
		# question encoding layers
		params = param_init_lstm(params, _concat(lstm_prefix_q, 1), EMBEDDINGS_DIM, LSTM_Q_OUT)
		params = param_init_lstm(params, _concat(lstm_prefix_q, 2), LSTM_Q_OUT, LSTM_Q_OUT)

		# history encoding layers
		params = param_init_lstm(params, _concat(lstm_prefix_h, 1), EMBEDDINGS_DIM, LSTM_H_OUT)
		params = param_init_lstm(params, _concat(lstm_prefix_h, 2), LSTM_H_OUT, LSTM_H_OUT)

		# decoding layers
		params = param_init_lstm(params, _concat(lstm_prefix_d, 1), EMBEDDINGS_DIM, LSTM_D_OUT)
		params = param_init_lstm(params, _concat(lstm_prefix_d, 2), LSTM_D_OUT, EMBEDDINGS_DIM)

	else:
		params = np.load(address)

	# initialize theano shared variables for params
	tparams = OrderedDict()
	for key, val in params.iteritems():
		print key, ': ', val.shape
		tparams[key] = theano.shared(val, name=key)

	return tparams

def build_encoder(tparams):
	'''
	Builds a computational graph for a memory network encoder
	'''
	global ff_prefix, lstm_prefix_q, lstm_prefix_h

	def _generate_memories(his, hmask):
		# generate embedding for one history
		hsteps = his.shape[0]
		out_1 = lstm_layer(tparams, his, _concat(lstm_prefix_h, 1), mask=hmask, n_steps=hsteps)

		in_2 = T.as_tensor_variable(out_1[0])

		out_2 = lstm_layer(tparams, in_2, _concat(lstm_prefix_h, 2), mask=hmask, n_steps=hsteps)

		return T.as_tensor_variable(out_2[0][-1])
	
	def _weigh_memories(mems, attention_vector):
		return (mems.T * attention_vector).T

	# image features extracted from vgg16
	img = T.matrix('img', dtype='float32')

	# steps x samples x embedding size
	que = T.tensor3('que', dtype='float32')

	# dialogs x max tokens x samples x embedding size
	his = T.tensor4('his', dtype='float32')
	
	# steps x samples
	qmask = T.matrix('qmask', dtype='float32')

	# dialogs x max tokens x samples
	hmask = T.tensor3('hmask', dtype='float32')
	
	qsteps = que.shape[0]
	memsize = his.shape[0]

	# encode questions
	out_1 = lstm_layer(tparams, que, _concat(lstm_prefix_q, 1), mask=qmask, n_steps=qsteps)
	
	in_2 = T.as_tensor_variable(out_1[0])

	out_2 = lstm_layer(tparams, que, _concat(lstm_prefix_q, 2), mask=qmask, n_steps=qsteps)

	qcode = out_2[0][-1]

	# generate the query vector
	in_3 = T.concatenate([img, qcode], axis=1)
	query = fflayer(tparams, in_3, _concat(ff_prefix, 1))

	# create memory
	mems, updates = theano.scan(_generate_memories, 
								sequences=[his, hmask],
								n_steps=memsize)

	mems = T.as_tensor_variable(mems)

	# compute attention
	attention_matrix = (mems * query).sum(axis=2)

	# to ensure convex combination, using softmax
	attention_matrix = T.nnet.softmax(attention_matrix.T)
	attention_matrix = attention_matrix.T

	# weighted memories
	mems_weighted, updates = theano.scan(_weigh_memories,
										 sequences=[mems, attention_matrix],
										 n_steps=memsize)

	memory = T.as_tensor_variable(mems_weighted).sum(axis=0)

	# passing through fully connected layer
	memvec = fflayer(tparams, memory,_concat(ff_prefix, 2))
	memNNcode = memvec + query

	return img, que, qmask, his, hmask, memNNcode

def build_decoder(tparams, code, max_steps):
	'''
	Build computational graph for a generative decoder
	'''

	global embeddings, word_idx_map, lstm_prefix_d

	def _decode_step(sbelow, sbefore_1, sbefore_2, cell_before_1, cell_before_2):
		'''
		Custom step for decoder, where the output of 2nd LSTM layer is fed into the 1st LSTM layer
		'''
		h_1, c_1 = lstm_layer(tparams, sbelow, _concat(lstm_prefix_d, 1), init_state=sbefore_1, init_memory=cell_before_1, one_step=True)

		h_2, c_2 = lstm_layer(tparams, h_1, _concat(lstm_prefix_d, 2), init_state=sbefore_2, init_memory=cell_before_2, one_step=True)

		return h_2, h_1, h_2, c_1, c_2
		
		
	def _softmax(inp):
		'''
		Chooses the right element from the outputs for softmax
		'''
		# for numerical stability of the output, a small value is added to all probabilities
		return T.nnet.softmax(T.dot(inp, embeddings)) + 1e-8

	n_samples = code.shape[0]
	hdim1 = code.shape[1]
	hdim2 = embeddings.shape[0]

	init_h1 = code
	init_h2 = T.alloc(0., n_samples, hdim2)
	# memory_1 = T.alloc(0., n_samples, hdim1)
	memory_1 = code
	memory_2 = T.alloc(0., n_samples, hdim2)

	init_token = T.tile(embeddings.T[word_idx_map['<sos>'], :], (n_samples, 1))
	
	# initial hidden state for both 1st layer is code
	tokens, updates = theano.scan(_decode_step,
									outputs_info=[init_token, init_h1, init_h2, memory_1, memory_2],
									n_steps=max_steps)

	tokens = T.as_tensor_variable(tokens[2])

	soft_tokens, updates = theano.scan(_softmax, sequences=tokens)

	return T.as_tensor_variable(soft_tokens)

# Complete graph
if len(sys.argv) > 2:
	print "Initializating parameters for model"
	tparams = initialize(sys.argv[2])
else:
	tparams = initialize()

# TRAINING
if len(sys.argv) <=1 or int(sys.argv[1]) == 0:
	print "Building encoder for the model"
	img, que, qmask, his, hmask, memcode = build_encoder(tparams)

	# printing value of encoder output
	memc_printed = theano.printing.Print('Encoded Value: ')(memcode)
	img_printed = theano.printing.Print('Input image: ')(img)
	que_printed = theano.printing.Print('Input question: ')(que)
	qmask_printed = theano.printing.Print('Question Mask: ')(qmask)
	his_printed = theano.printing.Print('Input history: ')(his)
	hmask_printed = theano.printing.Print('History mask: ')(hmask)
	
	# answer tensor should be a binary tensor with 1's at the positions which needs to be included
	# timesteps x number of answers in minibatch x vocabulary size
	ans = T.tensor3('ans', dtype='int8')

	print "Building decoder"
	pred = build_decoder(tparams, memcode, ans.shape[0])

	print "Building cost function"
	# cost function
	cost = (-T.log(pred) * ans).sum()

	inps = [img, que, qmask, his, hmask, ans]

	print "Constructing graph"
	f_cost = theano.function(inps, [cost, memc_printed, img_printed, que_printed, qmask_printed, his_printed, hmask_printed], on_unused_input='ignore', profile=False)
	print "Done!"

	print "Computing gradients"
	param_list=[val for key, val in tparams.iteritems()]
	grads = T.grad(cost, wrt=param_list)

	# computing norms
	f_grad_norm = theano.function(inps, [(g**2).sum() for g in grads], profile=False)
	f_weight_norm = theano.function([], [(v**2).sum() for k, v in tparams.iteritems()], profile=False)

	# gradients are clipped beyond certain values
	g2 = 0.
	for g in grads:
		g2 += (g**2).sum()
	new_grads = []
	for g in grads:
		new_grads.append(T.switch(g2 > (GRAD_CLIP**2),
									g / T.sqrt(g2)*GRAD_CLIP, g))
	grads = new_grads

	# learning rate
	lr = T.scalar(name='lr', dtype='float32')

	# gradients, update parameters
	print "Setting up optimizer"
	f_grad_shared, f_update = adam(lr, tparams, grads, inps, cost)

	# set learning rate before training
	lrate = learning_rate

	# time and cost will be output to the text file in BugReports folder
	if len(sys.argv) > 2:
		EPOCH_START = int(sys.argv[2].split('_')[-1].split('.')[0])
		training_output = open('../../BugReports/memory_network_train_output_' + str(reduced_instances) + '_' + str(learning_rate) + '.txt','a')
	else:
		EPOCH_START = 0
		training_output = open('../../BugReports/memory_network_train_output_' + str(reduced_instances) + '_' + str(learning_rate) + '.txt','w')

	for epoch in range(EPOCH_START, EPOCHS):
		train_data.reset()

		print 'Epoch ', epoch + 1
		epoch_cost = 0.0
		epoch_start = time.time()

		for batch_idx in range(train_data.batches):
			# ibatch, qbatch, mqbatch, hbatch, mhbatch, abatch = train_data.get_batch_lfe()
			# print 'ibatch:', ibatch.shape, 'qbatch:', qbatch.shape, 'hbatch:', hbatch.shape, 'abatch:', abatch.shape
			
			# batch_cost, memcode, i, q, qm, h, hm = f_cost(ibatch, qbatch, mqbatch, hbatch, mhbatch, abatch)

			t_start = time.time()
			# directly unfolds the tuple returned as arguments to the function which reduces memory footprint
			cost = f_grad_shared(*train_data.get_batch())
			f_update(lrate)
			td = time.time() - t_start

			epoch_cost += cost
			
			if not batch_idx % 100:
				training_output.write('Epoch: ' + str(epoch) + ', Batch ID: ' + str(batch_idx) + ', Cost: ' + str(cost) + ', Time: ' + str(td) + '\n')

		print 'Epoch:', epoch + 1, 'Cost:', epoch_cost, 'Time: ', time.time()-epoch_start
		training_output.write('Epoch: ' + str(epoch + 1) + ', Cost: ' + str(epoch_cost) + ', Time: ' + str(time.time()-epoch_start) + '\n')
		
		if (epoch + 1) % 5 == 0:
			print 'Saving... '

			params = {}
			for key, val in tparams.iteritems():
				params[key] = val.get_value()

			# numpy saving
			np.savez(MODEL_DIR + 'MemoryNetwork/memory_network_' + str(reduced_instances) + '_' + str(learning_rate) + '_' + str(epoch + 1)+'.npz', **params)
			np.savez(MODEL_DIR + 'Backup/memory_network_' + str(reduced_instances) + '_' + str(learning_rate) + '_' + str(epoch + 1)+'.npz', **params)			
			# np.save(MODEL_DIR + 'Backup/lfe_mask_' + str(reduced_instances) + '_' + str(learning_rate) + '_' + str(epoch + 1)+'.npy', params)
			print 'Done!'

		print 'Completed Epoch ', epoch + 1