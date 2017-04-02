import numpy as np

class data():
	def __init__(self, img, que, ans, ans_tokens, vocab_size, batch_size=128):
		self.img = img
		self.que = que
		self.ans = ans
		self.ans_tokens = ans_tokens
		self.batch_size = batch_size
		self.embed_size = que[0].shape[1]
		self.eos = que[0][-1,:]
		self.vocab_size = vocab_size
		self.eos_token = ans_tokens[0][-1]

		print 'Batch Size: ' + str(batch_size)
		print 'Embedding Size: ' + str(self.embed_size)


	# def get_counts(self):

	# def reset(self):

	# def get_batch(self):