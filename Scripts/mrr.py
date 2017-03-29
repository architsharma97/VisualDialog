'''
Simple script which produces the mean reciprocal rank given the reciprocal_ranks
'''
import numpy as np
import sys

# Argument 1: Rank file produced by the validation
reciprocal_ranks = np.asarray([1.0/float(r.split(',')[0]) for r in open(sys.argv[1],'r').read().splitlines()])
print reciprocal_ranks.mean()