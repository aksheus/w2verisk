import gensim.models
import numpy as np


class CompMatrix:
		
	def __init__(model):
		self.model = model
		#gensim.models.KeyedVectors.load_word2vec_format('C:\\Users\\abkma\\anlp\\GoogleNews-vectors-negative300.bin', binary=True)
 
	def get_matrix(bag):
		#return [[self.model.word_vec(word,use_norm=True)] for word in bag]
		pass

	def get_concat_matrix(bag1,bag2):
		pass
