from GetUserDict import UserDict
from ComputeMatrix import CompMatrix
from gensim.models.word2vec import Word2Vec,PathLineSentences
from gensim.models import KeyedVectors

if __name__ == '__main__':
    ud = UserDict()
	#gensim.models.KeyedVectors.load_word2vec_format('C:\\Users\\abkma\\anlp\\GoogleNews-vectors-negative300.bin', binary=True)
    pos_path = r'C:\Users\abkma\nlp\reddit-depression\cleaned-train\cleaned-pos\f=4'
    neg_path = r'C:\Users\abkma\nlp\reddit-depression\cleaned-train\cleaned-neg\chunk1-2-3-4-5-6-7-8-9-10'
    """sentences = PathLineSentences(neg_path)
    model = Word2Vec(sentences=sentences, size=100, alpha=0.025, window=5, min_count=2,negative=5)
    model.wv.save('neg_class.bin')
    print('success')"""
    pos_vectors = KeyedVectors.load('postvt_class.bin')
    neg_vectors = KeyedVectors.load('neg_class.bin')
    pos_comp_matrix = CompMatrix(pos_vectors)
    neg_comp_matrix = CompMatrix(neg_vectors)
    pos_matrix = pos_comp_matrix.get_matrix(ud.get_bag(pos_path))
    neg_matrix = neg_comp_matrix.get_matrix(ud.get_bag(neg_path))
    for key in neg_matrix.keys():
        print(key,'     ',neg_matrix[key])

