import sys
sys.path.append('../')

import numpy as np
import theano.tensor as tensor

from utils import *
from layers import get_layer, param_init_fflayer, fflayer, param_init_lstm, lstm_layer
from preprocess import preprocess

DATA_DIR='../../Data/'

# preprocess the training data to get input matrices and tensors
image_features, questions_tensor, answers_tensor=preprocess(DATA_DIR, load_dict=True, load_embedding_data=True, save_data=False)

# load the embeddings
embeddings=np.load(DATA_DIR+'embedding_matrix.npy')