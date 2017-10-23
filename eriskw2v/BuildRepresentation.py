from GetUserDict import UserDict 
import os
#from itertools import chain
import numpy as np

listdir = lambda z: os.listdir(z)
isfile = lambda z : os.path.isfile(z)
join = lambda v,w : os.path.join(v,w)

class Builder:

    def __init__(self,rep_matrix,pos_path,neg_path,test_path):
        self.rep_matrix = rep_matrix
        self.pos_files = (f for f in listdir(pos_path) if isfile(join(pos_path,f)))
        self.neg_files = (f for f in listdir(neg_path) if isfile(join(neg_path,f)))
        self.test_files = (f for f in listdir(test_path) if isfile(join(test_path,f)))
        self.ud = UserDict()
    
    def GetDocVector(self,document):
        bow = self.ud.get_bag(document)
        word_vectors = []
        for w in bow:
            try:
                word_vectors.append(self.rep_matrix[w])
            except KeyError:
                pass
        doc_matrix = np.stack(word_vectors)
        return np.mean(doc_matrix,axis=0)


