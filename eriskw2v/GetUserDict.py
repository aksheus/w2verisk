from nltk import word_tokenize
import os

join = lambda x,y: os.path.join(x,y)
isdir = lambda x: os.path.isdir(x)
isfile = lambda y: os.path.isfile(y)

class UserDict:

	def __init__(self,path):
		self.path = path
		self.files = (f for f in os.listdir(self.path) if isfile(join(self.path,f)))

	def get_user_data(self):
		user_data = {}
		for f in self.files:
			with open(join(self.path,f),errors='ignore') as read:
				all_words = word_tokenize(''.join(line for line in read ))
				all_words = [ w.lower() for w in all_words]
				user_data[f]=all_words
		return user_data

	def get_bag(self):
		bag = set()
		for f in self.files:
			with open(join(self.path,f),errors='ignore') as read:
				all_words = word_tokenize(''.join(line for line in read ))
				for w in all_words:
					bag.add(w.lower())
		return list(bag) 
		
	def combine_bags(self,bags):
		return set().union(*bags)


