import sys
sys.path.append('../')

import numpy as np
import theano.tensor as tensor

from utils import *
from layers import get_layer, param_init_fflayer, fflayer, param_init_lstm, lstm_layer
from preprocess import preprocess


