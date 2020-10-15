# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 07:42:19 2020

@author: NBOLLIG
"""
import numpy as np


def get_IMDB_dataset(seed = None):
    from tensorflow import keras as K 

    if seed == None:
        seed = np.asscalar(np.random.randint(2**32 - 1, size=1, dtype = 'int64'))

    # save np.load
    np_load_old = np.load
    
    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

    # Read in data as word index sequences
    (X, y), (X_holdout, y_holdout) = K.datasets.imdb.load_data(path="imdb.npz",
                                                      num_words=20000,
                                                      skip_top=0,
                                                      maxlen=None,
                                                      seed=seed,
                                                      start_char=1,
                                                      oov_char=2,
                                                      index_from=3)
    
    # restore np.load for future normal usage
    np.load = np_load_old
    
    # Padding
    maxlen = 100
    X = K.preprocessing.sequence.pad_sequences(X, maxlen=maxlen)
    X_holdout = K.preprocessing.sequence.pad_sequences(X_holdout, maxlen=maxlen)
    
    # Reshape data
    y = y.reshape((-1,1))
    y_holdout = y_holdout.reshape((-1,1))
    
    # Return data
    return X, y, X_holdout, y_holdout

def show_text(vector):
    from tensorflow import keras as K 
    
    word_to_id = K.datasets.imdb.get_word_index()
    
    word_to_id = {k:(v+3) for k,v in word_to_id.items()}
    word_to_id["<PAD>"] = 0
    word_to_id["<START>"] = 1
    word_to_id["<UNK>"] = 2
    word_to_id["<UNUSED>"] = 3
    
    id_to_word = {value:key for key,value in word_to_id.items()}
    
    words = []
    
    for id in vector:
        if id not in id_to_word:
            word = "<UNK>"
        else:
            word = id_to_word[id]
        
        words.append(word)
    
    return ' '.join(words)
