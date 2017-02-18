import numpy as np

class data():
	def __init__(self, img, que, ans, batch_size=128, maxlen=50):
		self.img = img
		self.que = que
		self.ans = ans
		self.batch_size = batch_size
		self.maxlen = maxlen

		self.get_counts()

	def get_counts(self):
		que_by_tokens = {}
		que_by_his_tokens= {}

		# account for the extra sos and eos symbols
		for img_idx in range(len(img)):

			# initialize history token count with the number of tokens in the caption
			token_count_his = len(ans[img_idx*11]) - 2

			for i in range(10):
				que_idx = img_idx * 10 + i
				
				qlen = self.que[que_idx].shape[0]
				if qlen in que_by_tokens:
					que_by_tokens[qlen].append(que_idx)
				else:
					que_by_tokens[qlen] = [que_idx]

				hislen = token_count_his + 2
				if hislen in que_by_his_tokens:
					que_by_his_tokens[hislen].append(que_idx)
				else:
					que_by_his_tokens[hislen] = [que_idx]

				# update
				ans_idx = img_idx * 11 + i + 1
				token_count_his += qlen + len(ans[ans_idx]) - 4

		self.que_by_tokens = que_by_tokens
		self.que_by_his_tokens = que_by_his_tokens