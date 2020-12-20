#!/usr/bin/python
# -*- coding: utf-8 -*-

import gensim
#from gensim.models.fasttext import FastText
import numpy as np

"""
CREATE WORD VECTORS
"""
def create_vectors(corpus_file, model_file, config, nb_channels):

    for i in range(nb_channels):
        print("Create vectors for channel", i+1)

        # USE GENSIM    				
        sentences = gensim.models.word2vec.LineSentence(corpus_file + "." + str(i))

        # sg defines the training algorithm. By default (sg=0), CBOW is used. Otherwise (sg=1), skip-gram is employed.
        model = gensim.models.Word2Vec(sentences=sentences, size=config["EMBEDDING_DIM"], window=config["WINDOW_SIZE"], min_count=config["MIN_COUNT"], sg=config["SG"], workers=8)
        #model = FastText(sentences=sentences, size=config["EMBEDDING_DIM"], window=config["WINDOW_SIZE"], min_count=config["MIN_COUNT"], workers=8, iter=1)

        # STORE MODEL INTO A .word2vec FILE
        f = open(model_file + ".word2vec" + str(i)  ,'w')
        vectors = []
        vector = '{} {}\n'.format(len(model.wv.index2word), config["EMBEDDING_DIM"])
        vectors.append(vector)
        f.write(vector)    
        for word in model.wv.index2word:
            vector = word + " " + " ".join(str(x) for x in model.wv[word]) + "\n"
            vectors.append(vector)
            f.write(vector)
        f.flush()
        f.close()

    """
    LOG
    """
    print("word2vec done.")
    
"""
GET W2VEC MODEL
"""
def get_w2v(vectors_file):
    return gensim.models.KeyedVectors.load_word2vec_format(vectors_file, binary=False)

"""
GET WORD VECTOR
"""
def get_vector(word, w2v):
    pass  
 
"""
FIND MOST SIMILAR WORD
"""
def get_most_similar(word, vectors_file):
    w2v = get_w2v(vectors_file)
    try:
        most_similar = w2v.most_similar(positive=[word])
    except:
        most_similar = []
    return most_similar


