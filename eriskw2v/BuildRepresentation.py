from GetUserDict import UserDict 
import os
#from itertools import chain
import numpy as np

listdir = lambda z: os.listdir(z)
isfile = lambda z : os.path.isfile(z)
join = lambda v,w : os.path.join(v,w)

class Builder:

    def __init__(self,rep_matrix,pos_path,neg_path,test_path,truth,pos_label,neg_label,dims):
        self.rep_matrix = rep_matrix
        self.pos_files = (join(pos_path,f) for f in listdir(pos_path) if isfile(join(pos_path,f)))
        self.neg_files = (join(neg_path,f) for f in listdir(neg_path) if isfile(join(neg_path,f)))
        self.test_files = (join(test_path,f) for f in listdir(test_path) if isfile(join(test_path,f)))
        self.ud = UserDict()
        self.truth = truth
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.features = [ 'dimension'+str(z+1) for z in range(dims) ]
        self.features.append('categories')

    def GetDocVector(self,document):
        bow = self.ud.get_bag_fromfile(document)
        word_vectors = []
        for w in bow:
            try:
                if w:
                    word_vectors.append(self.rep_matrix[w])
            except KeyError:
                pass
        doc_matrix = np.stack(word_vectors)
        return np.mean(doc_matrix,axis=0)

    def GetTrainRep(self,outfile):
        with open(outfile+'.csv','w',encoding='utf-8') as out:
            out.write(','.join(z for z in self.features))
            out.write('\n')
            for f in self.pos_files:
                dv = self.GetDocVector(f)
                out.write(','.join(str(v) for v in dv)+','+self.pos_label)
                out.write('\n')
            for f in self.neg_files:
                dv = self.GetDocVector(f)
                out.write(','.join(str(v) for v in dv)+','+self.neg_label)
                out.write('\n')
          
    def GetTruthDict(self):
        Truth = {}
        with open(self.truth) as read:
            for line in read:
                pieces = line.split()
                key = pieces[0].split('_')[-1]+'.txt'
                Truth[key]= pieces[-1]
        return Truth
    
    def GetTestRep(self,outfile):
        pass 



