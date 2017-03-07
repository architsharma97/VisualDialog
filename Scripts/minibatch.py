# NOTE: Remove <sos> and <eos> token when constructing history
import numpy as np

class data():
	def __init__(self, img, que, ans, ans_tokens, batch_size=128):
		self.img = img
		self.que = que
		self.ans = ans
		self.ans_tokens = ans_tokens
		self.batch_size = batch_size

		print 'Shape of image features: ', img.shape
		print 'Shape of questions_tensor: ', que.shape
		print 'Shape of answers_tensor: ', ans.shape
		print 'Shape of answers_tokens_idx: ', ans_tokens.shape
		print 'Batch Size: ' + str(batch_size)

	def get_counts(self):
		que_by_tokens = {}
		que_by_his_tokens= {}

		# account for the extra sos and eos symbols
		for img_idx in range(len(self.img)):

			# initialize history token count with the number of tokens in the caption
			token_count_his = self.ans[img_idx*11].shape[0] - 2

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
				token_count_his += qlen + self.ans[ans_idx].shape[0] - 4

		self.que_by_tokens = que_by_tokens
		self.que_by_his_tokens = que_by_his_tokens

		print "Counts for questions"
		for k,v in self.que_by_tokens.iteritems():
			print '%d : %d' %(k, len(v))
		
		print "Counts for history"
		for k,v in self.que_by_his_tokens.iteritems():
			print '%d : %d' %(k, len(v))

	# call only if you want to crash
	def process_for_lfe(self):
		print "Processing images"
		# repeating all the images 10 times
		self.img = np.asarray([np.repeat(feature, 10, axis=0) for feature in self.img])
		# reshaping
		self.img = self.img.reshape(-1, 4096)

		print "Creating history for each question"
		his = []
		for i in range(len(self.img)):
			if i%10 == 0:
				# the last question answer will never be used.
				ans_idx = i * 11 / 10
				prev = self.ans[ans_idx] 
			else:
				# i-1 is used for question and ans_idx for answers
				
				# removes eos from previous history
				prev_his = prev[ :prev.shape[0], :]
				
				# removes sos and eos tokens from previous question
				prev_que = self.que[i-1][1:self.que[i-1].shape[0], :]

				# removes sos token from the previous answer
				prev_ans = self.ans[ans_idx][1: , :]
				
				prev = np.concatenate((prev_his, prev_que, prev_ans), axis=0)
			
			his.append(prev)
			ans_idx += 1

		self.his = np.asarray(his)