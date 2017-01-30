# preprocesses data to provide trainable data
# refer to http://visualdialog.org/data
# not the most optimized files, lot of code in rudimentary python
import sys
sys.path.append("../")
import json

import numpy as np
import nltk
from utils import get_vgg16_features, get_embeddings

# requires the path of json file for training
def preprocess_train(path_to_data, save_dictionaries=False, save_image_features=False, save_embedding_matrix=False):
	'''
	path_to_data: Path from function call to the Data folder
	
	Example: '../../Data/'
	'''

	print "Loading JSON data file"
	data=json.load(open(path_to_data+'Training/visdial_0.5_train.json'))
	print "Preprocessing the dataset"
	# all images have 10 question-answer pairs in sequence
	image_ids=np.zeros((len(data,)))
	word_freq={}
	word_idx_map={}
	idx_word_map=[]

	# adding the standard tokens
	# end of sentence tokens
	word_idx_map["<eos>"]=0
	idx_word_map.append("<eos>")

	# unknown token
	word_idx_map["<unk>"]=1
	idx_word_map.append("<unk>")

	print "Collecting all tokens"
	for idx in range(len(data)):
		image_ids[idx]=int(data[idx]['image_id'])
		for token in nltk.word_tokenize(data[idx]['caption']):
			if token not in word_freq:
				word_freq[token]=1
			else:
				word_freq[token]+=1
		for dialog in data[idx]['dialog']:
			for token in nltk.word_tokenize(dialog['question']+" "+dialog['answer']):
				if token not in word_freq:
					word_freq[token]=1
				else:
					word_freq[token]+=1

	print "Mapping all tokens to index and vice versa"
	for token in word_freq:
		if word_freq[token]>=5:
			word_idx_map[token]=len(word_idx_map)
			idx_word_map.append(token)

	print "Dictionaries made!"
	print "Vocabulary size: " + str(len(word_idx_map))

	if save_dictionaries:
		print "Saving Dictionaries"
		np.save(path_to_data+"dictionary.npy", word_idx_map)
		np.save(path_to_data+"reverse_dictionary.npy", idx_word_map)

	# gets image features using the coco_ids
	image_features=get_vgg16_features(image_ids, path_to_data)

	if save_image_features:
		print "Saving image features for training set"
		np.save(path_to_data+"Training/train_image.npy",image_features)
	
	# embeddings
	embeddings=get_embeddings(word_idx_map, path_to_data+"Embeddings/glove.6B.300d.txt")

	# Since, the embeddings are pre-trained, both <eos> and <unk> map onto origin
	# To differentiate, the embedding for <eos> are set to random value
	embeddings[0]=np.random.rand(1,embeddings.shape(1))

	if save_embedding_matrix:
		print "Saving Embedding Matrix"
		np.save(path_to_data+"embedding_matrix.npy", embeddings)

	return image_ids, word_idx_map, idx_word_map