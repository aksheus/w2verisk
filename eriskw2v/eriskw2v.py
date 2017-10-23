from GetUserDict import UserDict
from ComputeMatrix import CompMatrix
from gensim.models.word2vec import Word2Vec,PathLineSentences
from gensim.models import KeyedVectors
from BuildRepresentation import Builder
import gc

if __name__ == '__main__':
    ud = UserDict()
	#gensim.models.KeyedVectors.load_word2vec_format('C:\\Users\\abkma\\anlp\\GoogleNews-vectors-negative300.bin', binary=True)
    pos_path = r'C:\Users\abkma\nlp\reddit-depression\cleaned-train\cleaned-pos\f=4'
    neg_path = r'C:\Users\abkma\nlp\reddit-depression\cleaned-train\cleaned-neg\chunk1-2-3-4-5-6-7-8-9-10'
    test_path = r'C:\Users\abkma\nlp\reddit-depression\cleaned-test\chunk1-2-3-4-5-6-7-8-9-10'
    truth = r'C:\Users\abkma\nlp\reddit-depression\test_golden_truth.txt'
    """sentences = PathLineSentences(neg_path)
    model = Word2Vec(sentences=sentences, size=100, alpha=0.025, window=5, min_count=2,negative=5)
    model.wv.save('neg_class.bin')
    print('success')"""
    pos_vectors = KeyedVectors.load('postvt_class.bin')
    neg_vectors = KeyedVectors.load('neg_class.bin')
    comp_matrix = CompMatrix()
    pos_matrix = comp_matrix.get_matrix(pos_vectors,ud.get_bag(pos_path))
    neg_matrix = comp_matrix.get_matrix(neg_vectors,ud.get_bag(neg_path))
    gc.collect()
    representation_matrix = comp_matrix.get_concat_matrix(pos_matrix,neg_matrix)
    """print(len(representation_matrix))
    count = 0
    for key in representation_matrix.keys():
        count+=1
        print(key,'     ',representation_matrix[key])
        print(len(representation_matrix[key]))
        if count > 5:
            break"""
    builder = Builder(representation_matrix,pos_path,neg_path,test_path,truth,'1','0',200)
    print(builder.GetTruthDict())
    #builder.GetTrainRep('train')
