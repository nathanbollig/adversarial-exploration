# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 09:37:51 2020

@author: NBOLLIG
"""

def exp1():
    """
    We assign nonzero probabilities to all amino acids in positive class - do this by 
    averaging distribution derived from motif (which does not have all aa) with the state 0 
    emission probability distribution.

    This introduces a hyperparameter k into the generative model and the mixture is
    (motif_distribution * k + state0_distribution)/(k+1).
    
    This experiment trains a LSTM on data generated using a range of k values, and reports the model accuracy.
    
    GOAL: Show task difficulty correlates with the signal to noise ratio.
    """
    pass

def exp2():
    """
    Vary p and run above type of experiment with fixed k.
    
    GOAL: Show task difficult correlates with the class prevalance.
    """
    pass

def exp3():
    """
    Run perturbation pipeline with no perturbation.
    """
    pass

def exp4():
    """
    Run perturbation pipeline with mutate 1 or k characters perturbation.
    """
    pass