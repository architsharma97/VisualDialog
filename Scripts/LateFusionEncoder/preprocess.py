# preprocesses data to provide trainable data
import json

def preprocess():
	FILE='../../Data/Training/visdial_0.5_train.json'
	data=json.load(open(FILE))

