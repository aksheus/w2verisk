import gensim.models
import numpy as np


class CompMatrix:
		
	def __init__(self,model):
		self.model = model
 
	def get_matrix(self,bag,norm=True):
		matrix = {}
		count = 0
		for word in bag:
			count+=1
			try:
				if word:
					matrix[word] = self.model.word_vec(word,use_norm=norm)
			except KeyError:
				pass
		return matrix

	def get_concat_matrix(bag1,bag2):
		pass
