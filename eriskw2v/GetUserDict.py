from nltk import word_tokenize
import os

join = lambda x,y: os.path.join(x,y)
isdir = lambda x: os.path.isdir(x)
isfile = lambda y: os.path.isfile(y)

class UserDict:

	def __init__(self):
		pass

	def get_user_data(self,path):
		user_data = {}
		files = (f for f in os.listdir(path) if isfile(join(path,f)))
		for f in files:
			with open(join(path,f),errors='ignore') as read:
				all_words = word_tokenize(''.join(line for line in read ))
				all_words = [ w.lower() for w in all_words]
				user_data[f]=all_words
		return user_data

	def get_bag(self,path):
		bag = set()
		files = (f for f in os.listdir(path) if isfile(join(path,f)))
		for f in files:
			with open(join(path,f),errors='ignore') as read:
				all_words = word_tokenize(''.join(line for line in read ))
				for w in all_words:
					bag.add(w.lower())
		return list(bag) 
		
	def combine_bags(self,bags):
		return set().union(*bags)


