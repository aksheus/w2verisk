import gensim.models
import numpy as np
import gc

class CompMatrix:
		
	def __init__(self):
		pass
 
	def get_matrix(self,model,bag,norm=False):
		matrix = {}
		count = 0
		for word in bag:
			count+=1
			try:
				if word:
					matrix[word] = model.word_vec(word,use_norm=norm)
			except KeyError:
				pass
		return matrix

	def get_concat_matrix(self,matrix1,matrix2):
		common_bag = list(set(matrix1.keys()).intersection(set(matrix2.keys())))
		concat_matrix = { word: np.concatenate((matrix1[word],matrix2[word]))
						  for word in common_bag }
		gc.collect()
		return concat_matrix
		