# -*- coding: utf-8 -*-
"""
Functions to compute perturbations of an animo acid sequence. They use the following standard
interface.

    input:
        seq - a list of indices
        y - original label for the sequence
        aa_vocab - a list of amino acid characters in the indexed ordering
        model - a model
        pertub_args - a dictionary
    output:
        a list of indices
        data - list of dictionaries with additional data, one per character flip

Generally we should have:
aa_vocab = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

Created on Tue Nov  3 09:49:33 2020
@author: NBOLLIG
"""
import random
from utils import decode_from_one_hot, encode_as_one_hot
from hotflip import one_flip
from keras import backend as K
from keras.layers import Input
import numpy as np

def no_perturb(seq, y, aa_vocab, model, generator):
    return seq, {}

def random_pt_mutations(seq, y, aa_vocab, model, generator, k):
    """
    Mutate k randomly-selected amino acids, to a random distinct character.
    """
    index_list = random.sample(list(range(len(seq))), k)
    
    for i in index_list:
        candidates = [a for a in list(range(len(aa_vocab))) if a not in [i]]
        j = random.choice(candidates)
        seq[i] = j
    
    return seq, {}

def hot_flip(seq, y, aa_vocab, model):
    """
    Perform HotFlip algorithm - flip the character.
    """
    seq = encode_as_one_hot(seq)
    seq, data = one_flip(model, seq, y)
    return decode_from_one_hot(seq), [data]

def derive_gradient_funct(model):
    """
    Return a gradient function that will be used to compute the gradient of the 
    loss function of the model at x,y with respect to inputs.
    
    Parameters:
        model - keras model from big_bang
        x - single one-hot-encoded sequence
        y - binary label
    """
    
    # Set up function to compute gradient
    y_true = Input(shape=(1,))
    ce = K.binary_crossentropy(y_true, model.output)
    grad_ce = K.gradients(ce, model.inputs)
    return K.function(model.inputs + [y_true], grad_ce)

def greedy_flip(seq, y, aa_vocab, model, generator=None, n_positions=60, n_characters=20, confidence_threshold = 0.5):
    """
    Greedily iterate hot flip until the predicted class label flips and the
    resulting prediction has confidence >= confidence_threshold.
    """
    seq = encode_as_one_hot(seq, n_positions=n_positions, n_characters=n_characters)
    pred = y
    conf = 0
    data = []
    init_pred_proba = model.predict(seq.reshape(1,n_positions, n_characters)).item()
    i = 1
    grad_function = derive_gradient_funct(model) # Only get the gradient from the keras model once to avoid mem leak
    
    while i < len(seq) and (int(y) == pred or conf < confidence_threshold):
        print('.', end='') # one dot per character flip
        seq, one_flip_data = one_flip(grad_function, seq, y, n_positions=n_positions, n_characters=n_characters)
        pred_proba = model.predict(seq.reshape(1, n_positions, n_characters)).item()
        pred = int(pred_proba > 0.5)
        
        if int(y) == 0:
            conf = pred_proba
        else:
            conf = 1 - pred_proba

        one_flip_data['pred_proba'] = pred_proba
        one_flip_data['conf'] = conf
        one_flip_data['init_pred_proba'] = init_pred_proba
        one_flip_data['change_number'] = i
        if generator != None:
            one_flip_data['actual_label'] = generator.predict(decode_from_one_hot(seq))
        data.append(one_flip_data)
        i += 1
    
    print('')
    return decode_from_one_hot(seq, n_positions=n_positions, n_characters=n_characters), data
    
    
    
    
    