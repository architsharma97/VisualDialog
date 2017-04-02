import numpy as np

class data():
	def __init__(self, img, que, ans, ans_tokens, vocab_size, batch_size=128):
		self.img = img
		self.que = que
		self.ans = ans
		self.ans_tokens = ans_tokens
		self.batch_size = batch_size
		self.embed_size = que[0].shape[1]
		self.eos = que[0][-1, :]
		self.vocab_size = vocab_size
		self.eos_token = ans_tokens[0][-1]

		print 'Batch Size: ' + str(batch_size)
		print 'Embedding Size: ' + str(self.embed_size)


	def get_counts(self):
		# get number of batches
		num_images = self.img.shape[0]
		self.batches = num_images/self.batch_size
		if num_images%self.batch_size:
			self.batches += 1

		print "Number of Batches: " + str(self.batches)
		
		self.img_order = np.asarray([i for i in range(num_images)])
		
		# get max lengths
		max_fact_length = np.zeros((num_images, ) dtype=np.int32)
		questions_length = np.zeros((self.que.shape[0], ), dtype=np.int32)
		answers_length = np.zeros((self.ans_tokens.shape[0], ), dtype=np.int32)
		
		# maximum length for a 'fact' for a given image
		for i in range(num_images):
			max_len = self.ans[i * 11].shape[0]

			for j in range(10):
				if j <= 9:
					max_len = max(max_len, self.que[i*10 + j].shape[0] + self.ans[i*11 + j + 1] - 2)
				
				questions_length[i*10 + j] = self.que[i*10 + j].shape[0]
				answers_length[i*10 + j] = self.ans_tokens[i*10 + j].shape[0]

			max_fact_length[i] = max_len

		self.max_fact_length = max_fact_length
		self.questions_length = questions_length
		self.answers_length = answers_length

	def reset(self):
		np.random.shuffle(self.img_order)
		self.curr = [0, 0]

	def get_batch(self):
		# get the range of images
		im_range = range(self.curr[0], min(self.curr[0] + self.batch_size, self.img.shape[0]))
		
		# get the image indices from the shuffled order
		im_idx = self.img_order[im_range]
		
		# question indices, use for answer tokens as well
		qu_idx = im_idx * 10 + self.curr[1]
		
		# embedded answer indices
		a_idx = im_idx * 11 + self.curr[1] + 1

		# max lengths
		max_batch_fact_length = self.max_fact_length[im_idx].max()
		max_batch_question_length = self.questions_length[qu_idx].max()
		max_batch_answer_length = self.answers_length[qu_idx].max()

		# retrieve old facts, if same set of images is being used
		if not self.curr[1]:
			facts = []
			facts_mask = []
		else:
			facts = self.facts
			facts_mask = self.facts_mask

		# final batch
		ibatch = self.img[im_idx].astype('float32')
		fact_embeddings = np.zeros((max_batch_fact_length, len(im_idx), self.embed_size), dtype=np.float32)
		fact_mask_array = np.zeros((max_batch_fact_length, len(im_idx)), dtype=np.float32)
		question_embeddings = np.zeros((max_batch_question_length, len(im_idx), self.embed_size), dtype=np.float32)
		question_mask = np.zeros((max_batch_question_length, len(im_idx)), dtype=np.float32)
		answer_batch = np.zeros((max_batch_answer_length, len(im_idx), self.vocab_size), dtype=np.int8)
		# answer_mask = np.zeros((max_batch_answer_length, len(im_idx)), dtype=np.int8)
		
		for idx in range(len(im_idx)):
			if self.curr[1] == 0:
				# construction of fact when the first question is asked
				cap_len = self.ans[a_idx[idx] - 1].shape[0]
				fact_embeddings[:cap_len, idx, :] = self.ans[a_idx[idx] - 1]
				fact_mask_array[:cap_len, idx] = 1.
			else:
				# append the previous QA pair along with mask
				# contains <eos>
				cur_len = self.que[qu_idx[idx] - 1].shape[0] - 1
				fact_embeddings[:cur_len, idx, :] = self.que[qu_idx[idx] - 1][:-1, :]
				
				# contains <sos>
				alen = self.ans[a_idx[idx] - 1].shape[0] - 1
				fact_embeddings[cur_len: cur_len + alen, idx, :] = self.ans[a_idx[idx] - 1][1:, :]
				cur_len += alen
				fact_mask_array[:cur_len, idx] = 1.
			
			# add the current question to the question matrix
			qlen = self.que[qu_idx[idx]].shape[0]
			question_embeddings[:qlen, idx, :] = self.que[qu_idx[idx]]	
			question_mask[:qlen, idx] = 1.

			# create the current answers
			atok_len = self.ans_tokens[qu_idx[idx]].shape[0]
			cur_ans = self.ans_tokens[qu_idx[idx]]
			# answer_mask[:atok_len, idx] = 1
			for j in range(atok_len):
				answer_batch[j, i, cur_ans[j]] = 1

		facts.append(fact_embeddings)
		facts_mask.append(fact_mask_array)

		self.facts = facts
		self.facts_mask = facts_mask

		# correct the indexing for the next batch 
		self.curr[1] += 1
		if self.curr[1] == 10:
			self.curr[1] = 0
			self.curr[0] = self.curr[0] + self.batch_size
			# reset should just after this exceeds number of images

		return ibatch, question_embeddings, question_mask, np.asarray(facts), \
				np.asarray(facts_mask), answer_batch