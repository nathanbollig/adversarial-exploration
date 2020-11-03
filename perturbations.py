# -*- coding: utf-8 -*-
"""
Functions to compute perturbations of an animo acid sequence. They accept as input
either a string or a list of indices, and output a list of indices.

Created on Tue Nov  3 09:49:33 2020

@author: NBOLLIG
"""
import numpy as np

def no_perturb(seq):
    # Convert to list of indices if input is a string
    if type(seq) == str:
        seq = self.string_to_list(seq)
            
    return seq