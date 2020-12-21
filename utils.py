# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 20:50:11 2020

@author: NBOLLIG
"""
import numpy as np
from keras.utils import to_categorical

def decode_from_one_hot(x):
    return np.argmax(x, axis=1, out=None).reshape(1, -1).tolist()[0]

def encode_as_one_hot(x):
    return to_categorical(x, num_classes=20).reshape(60,20)

def plot_aa_dist(pos, X_list, y_list, aa_vocab, class_label=None):
    """
    Plot the distribution of aa at a given position in the provided data.
    Optional restriction by class_label.
    """
    def get_data(pos, X_list, y_list, aa_vocab, class_label):
        X = np.concatenate(X_list)
        y = np.concatenate(y_list)
        
        indices = list(np.nonzero(y==class_label)[0])
        X = X[indices, :, :]
        
        dist = np.zeros((20,))
        
        for i in range(X.shape[0]):
            one_hot = X[i, pos]
            aa_index = np.nonzero(one_hot == 1)[0].item()
            dist[aa_index] += 1
            
        dist = dist / np.sum(dist)
        return dist
    
    import matplotlib.pyplot as plt
    if class_label == None:
        dist_0 = get_data(pos, X_list, y_list, aa_vocab, class_label=0)
        dist_1 = get_data(pos, X_list, y_list, aa_vocab, class_label=1)
        plt.plot(aa_vocab, dist_0, label="Class 0")
        plt.plot(aa_vocab, dist_1, label="Class 1")
        plt.legend(loc='lower left')
        plt.title("Class distributions")
    else:
        dist = get_data(pos, X_list, y_list, aa_vocab, class_label=class_label)
        plt.plot(aa_vocab, dist)
        plt.title("Distribution with class restriction = %i" % (class_label,))
    
    plt.xlabel("Amino Acid")
    plt.ylabel("Frequency at position %i" % (pos,))
    
    
    
        