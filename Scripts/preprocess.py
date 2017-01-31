# preprocesses data to provide trainable data
# refer to http://visualdialog.org/data
# not the most optimized files, lot of code in rudimentary python
import sys
sys.path.append("../")
import json

import numpy as np
import nltk
from utils import get_vgg16_features, get_embeddings, save_obj, load_obj

# requires the path of json file for training
def preprocess(path_to_data, 
	load_dict=False, 
	save_dictionaries=True, 
	load_embedding_matrix=False, 
	save_embedding_matrix=True, 
	save_data=True, 
	split='Train'):
	
	'''
	path_to_data: Path from function call to the Data folder
	load_*: load previously created dictionaries/embeddings. Automatically disbles save_*. Should be True for Validation/Testing.
	save_*: Save the embeddings/dictionaries created
	split: the data which is being processed. Can be 'Train', 'Val', 'Test'
	
	Example: '../../Data/'
	'''

	print "Loading JSON data file from split: " + str(split)
	
	if split=='Train':
		data=json.load(open(path_to_data+'Training/visdial_0.5_train.json'))
	elif split=='Val':
		# Vaidation split should not have to create vocabulary or embedding matrix
		data=json.load(open(path_to_data+'Validation/visdial_0.5_val.json'))
		load_dict=True
		load_embedding_matrix=True
	else:
		# Test split should not have to create vocabulary or embedding matrix
		data=json.load(open(path_to_data+"Test/visdial_0.5_test.json"))
		load_dict=True
		load_embedding_matrix=True
	
	print "Preprocessing the dataset"
	
	max_len_question=56
	max_len_answer=56

	# create a dictionary if not already available
	# to be used with only training data
	if not load_dict:
		print "Creating vocabulary for the dataset"
		word_freq={}
		word_idx_map={}
		idx_word_map=[]

		# token count in questions and answers for respective tensors
		ntokens_question=0
		ntokens_answer=0
		
		# adding the standard tokens
		# end of sentence tokens
		word_idx_map["<eos>"]=0
		idx_word_map.append("<eos>")

		# unknown token
		word_idx_map["<unk>"]=1
		idx_word_map.append("<unk>")

		print "Collecting all tokens"
		for idx in range(len(data)):
			for token in nltk.word_tokenize(data[idx]['caption']):
				if token not in word_freq:
					word_freq[token]=1
				else:
					word_freq[token]+=1
			for dialog in data[idx]['dialog']:
				
				tokens_ques=nltk.word_tokenize(dialog['question'])
				tokens_ans=nltk.word_tokenize(dialog['answer'])
				if len(tokens_ans) > ntokens_answer:
					ntokens_answer=len(tokens_ans)
				if len(tokens_ques) > ntokens_question:
					ntokens_question=len(tokens_ques)
				
				for token in tokens_ques+tokens_ans:
					if token not in word_freq:
						word_freq[token]=1
					else:
						word_freq[token]+=1

		print "Mapping all tokens to index and vice versa"
		for token in word_freq:
			if word_freq[token]>=5:
				word_idx_map[token]=len(word_idx_map)
				idx_word_map.append(token)
		
		max_len_question=ntokens_question+1
		max_len_answer=ntokens_answer+1

		print "Dictionaries made!"
		print "Vocabulary size: " + str(len(word_idx_map))
		print "Maximum number of tokens in a question: " + str(max_len_question)
		print "Maximum number of tokens in an answer: " + str(max_len_answer)

		if save_dictionaries:
			print "Saving Dictionaries"
			save_obj(word_idx_map, path_to_data+"dictionary.pkl")
			save_obj(idx_word_map, path_to_data+"reverse_dictionary.pkl")

	# load previously saved dictionary
	else:
		print "Loading dictionaries"
		word_idx_map=load_obj(path_to_data+"dictionary.pkl")
		idx_word_map=load_obj(path_to_data+"reverse_dictionary.pkl")

	# creates embedding matrix
	if not load_embedding_matrix:
		# embeddings
		embeddings=get_embeddings(word_idx_map, path_to_data+"Embeddings/glove.6B.300d.txt")

		# Since, the embeddings are pre-trained, both <eos> and <unk> map onto origin
		# To differentiate, the embedding for <eos> are set to random value
		embeddings[0]=np.random.rand(1,embeddings.shape[1])

		if save_embedding_matrix:
			print "Saving Embedding Matrix"
			np.save(path_to_data+"embedding_matrix.npy", embeddings)
	else:
		embeddings=np.load(path_to_data+"embedding_matrix.npy")

	# all images have 10 question-answer pairs in sequence
	image_ids=np.zeros((len(data),))
	questions_tensor=np.zeros((len(data)*10,max_len_question, embeddings.shape[1]))
	answers_tensor=np.zeros((len(data)*10, max_len_answer, embeddings.shape[1]))

	for idx in range(len(data)):
		image_ids[idx]=int(data[idx]['image_id'])


	# gets image features using the coco_ids
	image_features=get_vgg16_features(image_ids, path_to_data)

	if save_data:
		print "Saving data for " + split + " split"
		if split=='Train':
			np.save(path_to_data+"Training/train_image_features.npy", image_features)
		elif split=='Val':
			np.save(path_to_data+"Validation/val_image_features.npy", image_features)
		else:
			np.save(path_to_data+"Test/test_image_features.npy", image_features)
	else:
		return image_features, questions_tensor, answers_tensor
