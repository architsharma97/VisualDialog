# utility file
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np
from gensim.models import Word2Vec

# use model.predict(process_image(img_path)) to get the fc2 layer output
def vgg16():
	'''
	Returns the VGG16 model which outputs the 4096 dimensional embeddings
	of images. To be used with process_image.
	'''
	print "Building VGG16 model"
	base_model=VGG16(weights='imagenet',include_top=True)
	model=Model(input=base_model.input, output=base_model.get_layer('fc2').output)
	return model

def process_image(img_path):
	'''
	img_path: path to the actual image
	'''
	print "Pre-processing image"
	img = image.load_img(img_path, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	return x

# Typical path for glove embeddings if setup using default scripts
# '../../Data/Embeddings/glove.6B.300d.txt'
def get_embeddings(vocab, path, glove=True, EMBEDDING_SIZE=300):
	'''
	vocab: Given a dictionary of words, returns an embedding matrix.
		   Each word should be mapped to a unique index < len(vocab).
	glove: Defaults to text file format of glove embeddings.
		   Turn false for Word2Vec.
	path: Provide a path to the required embeddings.
	EMBEDDING_SIZE: Size of the embeddings in the described file. 
	Defaults to 300.
	'''

	embeddings=np.zeros((len(vocab),EMBEDDING_SIZE))

	# do not read in the whole embeddings file
	if glove:
		print "Constructing Embedding Matrix for given Vocabulary using GloVe pre-trained embeddings"
		with open(path) as embeddings_file:
			for line in embeddings_file:
				entry=line.split()
				if entry[0] in vocab:
					embeddings[vocab[entry[0]]]=np.asarray(entry[1:], dtype='float32')
	else:
		print "Constructing Embedding Matrix for given Vocabulary using Word2Vec embeddings"
		model=Word2Vec.load_word2vec_format(path,binary=True)
		for key, idx in vocab.iteritems():
			embeddings[idx]=model[key]
		# print "Cannot construct Embedding matrix from pre-trained Word2Vec"

	print "Constructed Embedding Matrix"
	return embeddings

