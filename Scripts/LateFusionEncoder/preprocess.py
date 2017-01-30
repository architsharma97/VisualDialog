# preprocesses data to provide trainable data
# refer to http://visualdialog.org/data
import sys
sys.path.append("../")
import json

import nltk
from utils import get_vgg16_features

def preprocess():
	FILE='../../Data/Training/visdial_0.5_train.json'
	data=json.load(open(FILE))

	# all images have 10 question-answer pairs in sequence
	image_ids=np.zeros((len(data,)))
	word_idx_map={}
	idx_word_map=[]

	# adding the tokens
	for idx in data:
		image_ids[idx]=int(data[idx]['image_id'])
		for tokens in nltk.word_tokenize(data[idx]['caption']):
			if tokens not in word_map:
				word_map[token]=1
			else:
				word_map[token]+=1
