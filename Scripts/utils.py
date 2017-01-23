# utility file for vgg16 feature extraction
# use model.predict(process_image(img_path)) to get the fc2 layer output
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np

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

def get_embeddings(vocab, glove=True, path, EMBEDDING_SIZE=300):
	'''
	vocab: Given a dictionary of words, returns an embedding matrix.
		   Each word should be mapped to a unique index < len(vocab).
	glove: Defaults to text file format of glove embeddings.
	path: Provide a path to the required embeddings.
	EMBEDDING_SIZE: Size of the embeddings in the described file. 
	Defaults to 300.
	'''

	print "Constructing the embedding matrix of Vocabulary"
	embeddings=np.zeros(len(vocab),)
