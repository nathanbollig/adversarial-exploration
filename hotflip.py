# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 20:10:53 2020

@author: NBOLLIG
"""
import numpy as np
from keras import backend as K
from keras.layers import Input
from utils import decode_from_one_hot

def compute_gradient(model, x, y):
    """
    Compute the gradient of the loss function of the model at x,y with respect to inputs.
    
    Parameters:
        model - keras model from big_bang
        x - single one-hot-encoded sequence
        y - binary label
    """
    
    # Set up function to compute gradient
    y_true = Input(shape=(1,))
    ce = K.binary_crossentropy(y_true, model.output)
    grad_ce = K.gradients(ce, model.inputs)
    func = K.function(model.inputs + [y_true], grad_ce)
    
    # Gradient of loss at (x,y) with respect to inputs
    return func([np.asarray(x).reshape((1,60,20)), y])[0][0]

def one_flip(model, x, y):
    """
    Compute a single character flip using the HotFlip algorithm.
    
    Parameters:
        model - keras model from big_bang
        x - single one-hot-encoded sequence
        y - binary label
    
    Returns:
        Perturbed (one-hot-encoded) sequence
        Loss increase associated with the flip
    """
    # Character sequence for x
    a_vector = decode_from_one_hot(x)
    
    # get gradient
    output = compute_gradient(model, x, y)
    
    # Find character flip that causes maximum increase in loss
    max_loss_increase = 0
    pos_to_change = None
    current_char_idx = None
    new_char_idx = None
    
    for i in range(60):
        a = a_vector[i]
        for b in range(20):
            loss_b = output[i][b]
            loss_a = output[i][a]
            loss_increase = loss_b - loss_a
            if loss_increase > max_loss_increase:
                max_loss_increase = loss_increase
                pos_to_change = i
                current_char_idx = a
                new_char_idx = b
    
    data = {}
    data['max_loss_increase'] = max_loss_increase
    data['pos_to_change'] = pos_to_change 
    data['current_char_idx'] = current_char_idx 
    data['new_char_idx'] = new_char_idx 
    
    x_perturb = np.copy(x).reshape((60,20))
    x_perturb[pos_to_change][current_char_idx] = 0
    x_perturb[pos_to_change][new_char_idx] = 1
    return x_perturb, data