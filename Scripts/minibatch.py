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

		self.get_counts()

	def get_counts(self):
		# for questions asked at different positions
		que_by_ans_tokens = [{} for i in range(10)]
		que_by_his_tokens= {}
		que_sizes = np.zeros((len(self.que), ), dtype=np.int16)
		his_sizes = np.zeros((len(self.que), ), dtype=np.int16)
		ans_sizes = np.zeros((len(self.que), ), dtype=np.int16)

		# accounts for the extra sos and eos symbols
		for img_idx in range(len(self.img)):

			# initialize history token count with the number of tokens in the caption
			token_count_his = self.ans[img_idx * 11].shape[0] - 2

			for i in range(10):
				que_idx = img_idx * 10 + i 
				ans_idx = img_idx * 11 + i + 1

				# adding size of question
				qlen = self.que[que_idx].shape[0]
				que_sizes[que_idx] = qlen
				

				# adding answer size
				alen = self.ans[ans_idx].shape[0]
				ans_sizes[que_idx] = self.ans[ans_idx].shape[0]

				# adding to dictionary
				if alen in que_by_ans_tokens[i]:
					que_by_ans_tokens[i][alen].append(que_idx)
				else:
					que_by_ans_tokens[i][alen] = [que_idx]

				hislen = token_count_his + 2
				# adding size of history
				his_sizes[que_idx] = hislen

				# adding to dictionary
				if hislen in que_by_his_tokens:
					que_by_his_tokens[hislen].append(que_idx)
				else:
					que_by_his_tokens[hislen] = [que_idx]

				# update token length of history
				token_count_his += qlen + alen - 4
				

		self.que_by_ans_tokens = que_by_ans_tokens
		self.que_by_his_tokens = que_by_his_tokens
		self.que_sizes = que_sizes
		self.his_sizes = his_sizes
		self.ans_sizes = ans_sizes

		# questions are binned at two levels
		# 1) The index at which they were asked with respect to the image
		# 2) The number of tokens in the questions
		print "Counts for tokens in answers"
		for i in range(10):
			print "for questions asked at position " + str(i + 1)
			for k,v in self.que_by_ans_tokens[i].iteritems():
				print '%d : %d' %(k, len(v))
		
		print "Counts for history"
		for k,v in self.que_by_his_tokens.iteritems():
			print '%d : %d' %(k, len(v))

		print "Computing number of batches"
		self.batches = 0
		# batches are constructed such that question lengths are same
		for i in range(10):
			for key, val in self.que_by_ans_tokens[i].iteritems():
				if (len(val)) % (self.batch_size):
					self.batches += len(val)/self.batch_size + 1
				else:
					self.batches += len(val)/self.batch_size

		print 'Number of Batches: ' + str(self.batches)

		# getting a list of allowed tokens in questions
		self.qlen_order = []
		for i in range(10):
			self.qlen_order += [key for key, val in self.que_by_ans_tokens[i].iteritems()]
		
		# removing redundant entries
		self.qlen_order = set(self.qlen_order)
		self.qlen_order = list(self.qlen_order)

	# pretty sure to crash
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

	def reset(self):
		np.random.shuffle(self.qlen_order)
		for i in range(10):
			for key, val in self.que_by_ans_tokens[i].iteritems():
				np.random.shuffle(self.que_by_ans_tokens[i][key])

		# question ordered wrt image, index in qlen_order, index in the list of questions associated with previous two
		self.curr = [0, 0, 0]
		while self.qlen_order[self.curr[1]] not in self.que_by_ans_tokens[self.curr[0]]:
			self.curr[0] += 1

	def get_batch_lfe(self):
		# getting number of tokens in the question matrix
		que_order_idx = self.curr[0]
		ans_tokens = self.qlen_order[self.curr[1]]

		# checks if enough questions for batch size, else makes a smaller batch
		if len(self.que_by_ans_tokens[que_order_idx][ans_tokens][self.curr[2]:]) > self.batch_size:
			qidx = self.que_by_ans_tokens[que_order_idx][ans_tokens][self.curr[2]: self.curr[2]+self.batch_size]
			self.curr[2] += self.batch_size
		else:
			qidx = self.que_by_ans_tokens[que_order_idx][ans_tokens][self.curr[2]:]
			que_order_idx += 1
			while que_order_idx < 10 and (ans_tokens not in self.que_by_ans_tokens[que_order_idx]):
				que_order_idx += 1

			if que_order_idx < 10:
				self.curr = [que_order_idx, self.curr[1], 0]
			else:
				self.curr = [0, self.curr[1] + 1, 0]
				while self.curr[1] < len(self.qlen_order) and self.qlen_order[self.curr[1]] not in self.que_by_ans_tokens[self.curr[0]]:
					self.curr[0] += 1
		
		# first pass to get maximum sizes of history and question sizes
		mhsize = 0
		mqsize = 0
		for idx in qidx:
			if self.his_sizes[idx] > mhsize:
				mhsize = self.his_sizes[idx]
			if self.que_sizes[idx] > mqsize:
				mqsize = self.que_sizes[idx]
		
		# answer tokens does not need the start token as the first token in predicted answer will not be start token. 
		ibatch = np.zeros((len(qidx), self.img.shape[1]), dtype=np.float32)
		qbatch = np.zeros((mqsize, len(qidx), self.embed_size), dtype=np.float32)
		mqbatch = np.zeros((mqsize, len(qidx)), dtype=np.float32)
		hbatch = np.zeros((mhsize, len(qidx), self.embed_size), dtype=np.float32)
		mhbatch = np.zeros((mhsize, len(qidx)), dtype=np.float32)
		abatch = np.zeros((ans_tokens - 1, len(qidx), self.vocab_size),  dtype=np.int8)

		for i, idx in enumerate(qidx):
			# question batch and mask
			qlen = self.que[idx].shape[0]
			qbatch[:qlen, i, :] = self.que[idx]
			mqbatch[:qlen, i] = 1.

			# image batch
			ibatch[i, :] = self.img[idx/10, :]
			# answer index
			ans_idx = (idx/10) * 11 + idx % 10 + 1

			# construction of answer
			cur_ans = self.ans_tokens[idx]
			for j in range(len(cur_ans)):
				abatch[j, i, cur_ans[j]] = 1

			# contruction of history batch
			cur_len = self.ans[(idx/10)*11].shape[0] - 1
			# append caption to history
			hbatch[:cur_len, i, :] = self.ans[(idx/10)*11][:-1,:]
			for j in range((idx/10)*10, idx):
				# append question to history
				qclen = self.que[j].shape[0] - 2
				hbatch[cur_len:cur_len+qclen, i, :] = self.que[j][1:-1, :]
				cur_len += qclen
				
				# append answer to history
				ans_idx = (j/10) * 11 + j % 10 + 1
				aclen = self.ans[ans_idx].shape[0] - 2
				hbatch[cur_len:cur_len+aclen,i, :] = self.ans[ans_idx][1:-1, :]
				cur_len += aclen
			hbatch[cur_len, i, :] = self.eos
			# create mask for history
			mhbatch[:cur_len + 1, i] = 1.

		return ibatch, qbatch, mqbatch, hbatch, mhbatch, abatch
	
	def get_batch_seq2seq(self):
		# getting number of tokens in the question matrix
		que_order_idx = self.curr[0]
		ans_tokens = self.qlen_order[self.curr[1]]

		# checks if enough questions for batch size, else makes a smaller batch
		if len(self.que_by_ans_tokens[que_order_idx][ans_tokens][self.curr[2]:]) > self.batch_size:
			qidx = self.que_by_ans_tokens[que_order_idx][ans_tokens][self.curr[2]: self.curr[2]+self.batch_size]
			self.curr[2] += self.batch_size
		else:
			qidx = self.que_by_ans_tokens[que_order_idx][ans_tokens][self.curr[2]:]
			que_order_idx += 1
			while que_order_idx < 10 and (ans_tokens not in self.que_by_ans_tokens[que_order_idx]):
				que_order_idx += 1

			if que_order_idx < 10:
				self.curr = [que_order_idx, self.curr[1], 0]
			else:
				self.curr = [0, self.curr[1] + 1, 0]
				while self.curr[1] < len(self.qlen_order) and self.qlen_order[self.curr[1]] not in self.que_by_ans_tokens[self.curr[0]]:
					self.curr[0] += 1

		mqsize = 0
		for idx in qidx:
			if self.que_sizes[idx] > mqsize:
				mqsize = self.que_sizes[idx]

		qbatch = np.zeros((mqsize, len(qidx), self.embed_size), dtype=np.float32)
		qmask = np.zeros((mqsize, len(qidx)), dtype=np.float32)
		abatch = np.zeros((ans_tokens - 1, len(qidx), self.vocab_size),  dtype=np.int8)

		for i, idx in enumerate(qidx):
			# question batch and mask
			qlen = self.que[idx].shape[0]
			qbatch[:qlen, i, :] = self.que[idx]
			qmask [:qlen, i] = 1.
			ans_idx = (idx/10)*11 + idx%10 + 1

			# construction of answer
			cur_ans = self.ans_tokens[idx]
			for j in range(len(cur_ans)):
				abatch[j, i, cur_ans[j]] = 1
				
		return qbatch, qmask, abatch
