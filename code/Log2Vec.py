import argparse
from gensim.models.word2vec import Word2Vec
import gensim
import numpy as np

def cos( vector1, vector2):
        return float(np.sum(vector1*vector2))/(np.linalg.norm(vector1)*np.linalg.norm(vector2))

def load_model(filename,is_binary=False):
    model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary = is_binary)
    return model

def getLogVector(para):
    template_file = para['template_file']
    model = load_model(para['word_model'])
    dimension = para['dimension']
    template_vector_file = para['template_vector_file']
    template_to_index = {}
    index_to_template = {}
    template_to_vector = {}
    template_num = 0
    with open(template_file) as IN:
        for line in IN:
                template_num += 1
    f = open(template_vector_file, 'w')
    f.writelines(str(template_num)+' '+str(para['dimension'])+'\n') #word2vec的模型格式，第一行为单词数&维度
    index = 1
    with open(template_file) as IN:
        for line in IN:
            template = line.strip()
            l = template.split()
            cur_vector = np.zeros(dimension)
            for word in l:
                cur_vector += model[word]
            cur_vector /= len(l)
            template_to_vector[template] = cur_vector
            template_to_index[template] = str(index)
            index_to_template[index] = template
            f.writelines(str(index))
            for v in cur_vector:
                f.writelines(' '+str(v))
            f.writelines('\n')
            index += 1
            
    return (template_to_index, index_to_template, template_to_vector)


import os
from gensim.models.word2vec import Word2Vec
import gensim
import numpy as np
class Log2Vec:
    def __init__(self, model_file, is_binary=False):
        print('reading log2vec model')
        model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary = is_binary)
    
        self.model = model
        self.dimension = len(model['1'])
        print(' Log2Vec.dimension:', self.dimension)

    def word_to_most_similar(self, y_word, topn = 1):
        return self.model.most_similar(positive = y_word,topn = topn)
    
    def vector_to_most_similar(self, y_vector, topn = 1):
        temp_dict = {}
        for t in self.vector_template_tuple:
            template_index = t[1]
            vector = t[0]
            temp_dict[template_index] = self.cos(y_vector, vector)
        sorted_final_tuple=sorted(temp_dict.items(),key=lambda asd:asd[1] ,reverse=True)
        return sorted_final_tuple[:topn] 
    
    
    def cos(self, vector1, vector2):
        return float(np.sum(vector1*vector2))/(np.linalg.norm(vector1)*np.linalg.norm(vector2))
        
        
        
    def get_cosine_matrix(self, _matrixB):
        _matrixA = self.template_matrix
        _matrixA_matrixB = _matrixA * _matrixB.reshape(len(_matrixB),-1)
        _matrixA_norm = np.sqrt(np.multiply(_matrixA,_matrixA).sum())
        _matrixB_norm = np.sqrt(np.multiply(_matrixB,_matrixB).sum())
        return np.divide(_matrixA_matrixB, _matrixA_norm * _matrixB_norm.transpose())

    def vector_to_most_similar_back(self, vectorB, topn = 1):
        cosine_matrix = self.get_cosine_matrix(vectorB)
        sort_dict = {}
        for i, sim in enumerate(cosine_matrix):
            template_num = str(i+1)
            sort_dict[template_num] = sim
        sorted_final_tuple=sorted(sort_dict.items(),key=lambda asd:asd[1] ,reverse=True)
        return sorted_final_tuple[:topn] 
        
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-logs', help='log file', type=str, default='./data/BGL_without_variables.log')
    parser.add_argument('-word_model', help='word_model', type=str, default='./middle/bgl_words.model')
    parser.add_argument('-log_vector_file', help='template_vector_file', type=str, default='./middle/bgl_log.vector')
    #parser.add_argument('-template_num', help='template_num', type=int, default=373)
    parser.add_argument('-dimension', help='dimension', type=int, default=32)
    args = parser.parse_args()

    para = {
        'template_file' : args.logs,
        'word_model': args.word_model,
        'template_vector_file': args.log_vector_file,
        #'template_num':args.template_num,
        'dimension':args.dimension
    }
    print('log input:', args.logs)
    print('word vectors input:', args.word_model)
    print('log vectors output:', args.log_vector_file)
    getLogVector(para)
    print('end~~')
