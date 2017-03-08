# NOTE: Remove <sos> and <eos> token when constructing history
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

	def get_counts(self):
		que_by_tokens = {}
		que_by_his_tokens= {}
		que_sizes = np.zeros((len(self.que), ), dtype=np.int64)
		his_sizes = np.zeros((len(self.que), ), dtype=np.int64)
		ans_sizes = np.zeros((len(self.que), ), dtype=np.int64)

		# account for the extra sos and eos symbols
		for img_idx in range(len(self.img)):

			# initialize history token count with the number of tokens in the caption
			token_count_his = self.ans[img_idx*11].shape[0] - 2

			for i in range(10):
				que_idx = img_idx * 10 + i 

				qlen = self.que[que_idx].shape[0]
				# adding size of question
				que_sizes[que_idx] = qlen

				# adding to dictionary
				if qlen in que_by_tokens:
					que_by_tokens[qlen].append(que_idx)
				else:
					que_by_tokens[qlen] = [que_idx]

				hislen = token_count_his + 2
				# adding size of history
				his_sizes[que_idx] = hislen

				# adding to dictionary
				if hislen in que_by_his_tokens:
					que_by_his_tokens[hislen].append(que_idx)
				else:
					que_by_his_tokens[hislen] = [que_idx]

				# update
				ans_idx = img_idx * 11 + i + 1
				token_count_his += qlen + self.ans[ans_idx].shape[0] - 4
				
				# adding answer size
				ans_sizes[que_idx] = self.ans[ans_idx].shape[0]

		self.que_by_tokens = que_by_tokens
		self.que_by_his_tokens = que_by_his_tokens
		self.que_sizes = que_sizes
		self.his_sizes = his_sizes
		self.ans_sizes = ans_sizes

		print "Counts for questions"
		for k,v in self.que_by_tokens.iteritems():
			print '%d : %d' %(k, len(v))
		
		print "Counts for history"
		for k,v in self.que_by_his_tokens.iteritems():
			print '%d : %d' %(k, len(v))

		print "Computing number of batches"
		self.batches = 0
		# batches are constructed such that question lengths
		for key, val in self.que_by_tokens.iteritems():
			if not len(val)%self.batch_size:
				self.batches += len(val)/self.batch_size + 1
			else:
				self.batches += len(val)/self.batch_size

		# getting a list of allowed tokens in questions
		self.qlen_order = [key for key, val in self.que_by_tokens.iteritems()]
	
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

	def reset(self):
		np.random.shuffle(self.qlen_order)
		for key, val in self.que_by_tokens.iteritems():
			np.random.shuffle(self.que_by_tokens[key])

		self.curr = [0, 0]

	def get_batch(self):
		# getting number of tokens in the question matrix
		que_tokens = self.qlen_order[self.curr[0]]

		# checks if enough questions for batch size, else makes a smaller batch
		if len(self.que_by_tokens[que_tokens][self.curr[1]:]) > self.batch_size:
			qidx = self.que_by_tokens[que_tokens][self.curr[1]: self.curr[1]+self.batch_size]
			self.curr[1] += self.batch_size
		else:
			qidx = self.que_by_tokens[que_tokens][self.curr[1]:]
			self.curr = [self.curr[0] + 1, 0]
		
		# first pass to get maximum sizes of history and answers
		mhsize = 0
		masize = 0
		for idx in qidx:
			if self.his_sizes[idx] > mhsize:
				mhsize = self.his_sizes[idx]
			if self.ans_sizes[idx] > masize:
				masize = self.ans_sizes[idx]
		
		ibatch = np.zeros((len(qidx), self.img.shape[1])).astype('float32')
		qbatch = np.zeros((que_tokens, len(qidx), self.embed_size)).astype('float32')
		hbatch = np.tile(self.eos, (mhsize, len(qidx), 1)).astype('float32')
		abatch = np.zeros((masize, len(qidx), self.vocab_size)).astype('int64')

		for i, idx in enumerate(qidx):
			qbatch[:, i, :] = self.que[idx]
			ibatch[i, :] = self.img[idx/10, :]
			ans_idx = (idx/10)*11 + idx%10 + 1

			# construction of answer
			cur_ans = self.ans_tokens[idx]
			for j in range(len(cur_ans)):
				abatch[j, i, cur_ans[j]] = 1

			for j in range(len(cur_ans), masize):
				abatch[j, i, self.eos_token] = 1

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
				ans_idx = (j/10)*11 + j%10 + 1
				aclen = self.ans[ans_idx].shape[0] - 2
				hbatch[cur_len:cur_len+aclen,i, :] = self.ans[ans_idx][1:-1, :]
				cur_len += aclen

		return ibatch, qbatch, hbatch, abatch