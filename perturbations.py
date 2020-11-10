# -*- coding: utf-8 -*-
"""
Functions to compute perturbations of an animo acid sequence. They use the following standard
interface.

    input:
        seq - a list of indices
        aa_vocab - a list of amino acid characters in the indexed ordering
    output:
        a list of indices.

Generally we should have:
aa_vocab = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

Created on Tue Nov  3 09:49:33 2020
@author: NBOLLIG
"""
import random



def no_perturb(seq, aa_vocab):
    return seq

def random_pt_mutations(seq, aa_vocab, k):
    """
    Mutate k randomly-selected amino acids, to a random distinct character.
    """
    index_list = random.sample(list(range(len(seq))), k)
    
    for i in index_list:
        candidates = [a for a in list(range(len(aa_vocab))) if a not in [i]]
        j = random.choice(candidates)
        seq[i] = j
    
    return seq