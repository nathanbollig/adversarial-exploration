# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 20:10:53 2020

@author: NBOLLIG
"""
import numpy as np
from utils import decode_from_one_hot

def evaluate_grad_funct(f, x, y, n_positions=60, n_characters=20):
    # Gradient of loss at (x,y) with respect to inputs
    return f([np.asarray(x).reshape((1,n_positions,n_characters)), y])[0][0]

def one_flip(gradient_func, x, y, n_positions=60, n_characters=20, ignore_char_indices=[-1], weights=None, gamma=0.1, cost=100, aa_vocab=None):
    """
    Compute a single character flip using the HotFlip algorithm.
    
    Parameters:
        gradient_func - keras function for gradient computation on model
        x - single one-hot-encoded sequence
        y - binary label
        weights - if provided, then flip is randomized according to provided weights; it is a dict like MatrixInfo.blosum62
        aa_vocab - provided if weights are provided
        gamma - provided if weights are provided
    
    Returns:
        Perturbed (one-hot-encoded) sequence
        Loss increase associated with the flip
    """
    # Character sequence for x
    x = np.array(x).reshape((n_positions, n_characters))
    a_vector = decode_from_one_hot(x, n_positions=n_positions, n_characters=n_characters)
    
    # get gradient
    output = evaluate_grad_funct(gradient_func, x, y, n_positions=n_positions, n_characters=n_characters)
    
    if weights == None:
        # Find character flip that causes maximum increase in loss
        max_loss_increase = 0
        pos_to_change = None
        current_char_idx = None
        new_char_idx = None
        
        for i in range(n_positions):
            a = a_vector[i]
            for b in range(n_characters):
                if ignore_char_indices != None:
                    if a in ignore_char_indices:
                        continue
                loss_b = output[i][b]
                loss_a = output[i][a]
                loss_increase = loss_b - loss_a
                if loss_increase > max_loss_increase:
                    max_loss_increase = loss_increase
                    pos_to_change = i
                    current_char_idx = a
                    new_char_idx = b
    else:        
        assert(aa_vocab != None)
        
        # Initialize distribution to sample from
        D = np.zeros((n_positions, n_characters))
        
        # Fill in with weighted loss increases
        for i in range(n_positions):
            a = a_vector[i]
            for b in range(n_characters):
                if ignore_char_indices != None:
                    if a in ignore_char_indices:
                        continue
                
                # Get estimated loss increase
                loss_b = output[i][b]
                loss_a = output[i][a]
                loss_increase = loss_b - loss_a
                
                # Get weight from blossum
                char_a = aa_vocab[a]
                char_b = aa_vocab[b]
                
                def score_match(pair, matrix):
                    if pair in matrix:
                        return matrix[pair]
                    elif (tuple(reversed(pair))) in matrix:
                        return matrix[(tuple(reversed(pair)))]
                    else:
                        return 0
                
                pair = (char_a, char_b)
                weight = np.exp(gamma * score_match(pair, weights))
                
                # Update D
                D[i][b] = weight * loss_increase
        
        # Apply cost to scale before softmax
        D_before = D
        D = cost * D     
        
        # Subtract max from D before applying softmax - does not change result but improves num stability
        D = D - np.max(D)
        
        # Softmax
        D = np.exp(D) / np.sum(np.exp(D))
        
        # Sample
        D_flattened = np.ravel(D)
        sample_idx = np.random.choice(len(D_flattened), p=D_flattened)
        pos_to_change, new_char_idx = np.unravel_index(sample_idx, D.shape)
        current_char_idx = a_vector[pos_to_change]
        max_loss_increase = output[pos_to_change][new_char_idx] - output[pos_to_change][current_char_idx]

    data = {}
    data['max_loss_increase'] = max_loss_increase
    data['pos_to_change'] = pos_to_change 
    data['current_char_idx'] = current_char_idx 
    data['new_char_idx'] = new_char_idx
    
    x_perturb = np.copy(x).reshape((n_positions, n_characters))
    x_perturb[pos_to_change][current_char_idx] = 0
    x_perturb[pos_to_change][new_char_idx] = 1
    return x_perturb, data