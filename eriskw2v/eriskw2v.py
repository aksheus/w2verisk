from GetUserDict import UserDict
from ComputeMatrix import CompMatrix
from gensim.models.word2vec import Word2Vec,PathLineSentences
from gensim.models import KeyedVectors

if __name__ == '__main__':
    ud = UserDict()
	#gensim.models.KeyedVectors.load_word2vec_format('C:\\Users\\abkma\\anlp\\GoogleNews-vectors-negative300.bin', binary=True)
    pos_paths = 'C:\\Users\\abkma\\nlp\\reddit-depression\\cleaned-train\\cleaned-pos\\chunk1-2-3-4-5-6-7-8-9-10'
    """             'C:\\Users\\abkma\\nlp\\reddit-depression\\cleaned-train\\cleaned-pos\\chunk1-2-3-4',
                 'C:\\Users\\abkma\\nlp\\reddit-depression\\cleaned-train\\cleaned-pos\\chunk1-2-3',
                 'C:\\Users\\abkma\\nlp\\reddit-depression\\cleaned-train\\cleaned-pos\\chunk1-2',
                 'C:\\Users\\abkma\\nlp\\reddit-depression\\cleaned-train\\cleaned-pos\\chunk_1']"""
    neg_path = r'C:\Users\abkma\nlp\reddit-depression\cleaned-train\cleaned-neg\chunk1-2-3-4-5-6-7-8-9-10'
    #sentences = PathLineSentences(pos_paths)
    #model = Word2Vec(sentences=sentences, size=100, alpha=0.025, window=5, min_count=2,negative=5)
    #model.wv.save('neg_class.bin')
    word_vectors = KeyedVectors.load('pos_class.bin')
    comp = CompMatrix(word_vectors)
    neg_matrix = comp.get_matrix(ud.get_bag(pos_paths))
    for key in neg_matrix.keys():
        print(key,'     ',neg_matrix[key])

