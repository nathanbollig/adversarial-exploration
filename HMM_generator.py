# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 08:38:21 2020

@author: NBOLLIG
"""
import numpy as np
from hmmlearn import hmm
    
class HMMGenerator():
    def __init__(self, seq_length = 60, start = 25, active_site_length = 10):
        self.seq_length = seq_length
        self.start = start
        self.active_site_length = active_site_length

        self.set_aa_emissions()
        self.set_parameters()
        self.set_model()
    
    def set_model(self):
        self.model = hmm.MultinomialHMM(n_components = self.n_components)
        self.model.startprob_ = self.startprob
        self.model.transmat_ = self.transmat
        self.model.emissionprob_ = self.emissionprob
    
    def generate_one_sequence(self):
        x, path = self.model.sample(n_samples = self.seq_length)
        x = x.reshape(1, -1).tolist()[0]
        
        if self.start in path:
            y = 0
        else:
            y = 1
        
        return x, y       
    
    def generate(self, n_samples=100):
        """
        Generates list of sequences X and list of labels y.
        
        Returns:
            X - list of sequences of length seq_length
            y - list of labels (0 or 1)
    
        """
        X = []
        Y = []
        for i in range(n_samples):
            x, y = self.generate_one_sequence()
            X.append(x)
            Y.append(y)
        
        # Convert y to an array
        Y = np.array(Y, dtype=int)
        
        return X, Y

    
    def set_aa_emissions(self):
        """
        Use approximately the data from [Tian, 2016] for background and state 0 emission profiles. 
        Use the motif KRSFIEDLLFNKV for state 1 emission profile [Robson, 2020].
        """
        
        self.aa_list = list('ARNDCEQGHILKMFPSTWYV')
        self.background_emission = [0.075, 0.042, 0.045, 0.059, 0.033, 0.057, 0.037, 0.074, 0.039, 0.038, 0.076, 0.074, 0.019, 0.04, 0.05, 0.081, 0.062, 0.013, 0.034, 0.052]
        self.state0_emission = [0.06, 0.042, 0.095, 0.088, 0.017, 0.089, 0.031, 0.064, 0.07, 0.013, 0.017, 0.021, 0.018, 0.031, 0.03, 0.063, 0.04, 0.103, 0.099, 0.009]
        
        motif = list('KRSFIEDLLFNKV')
        self.state1_emission = np.zeros((20,))
        for aa in motif:
            self.state1_emission[self.aa_list.index(aa)] += 1 / len(motif)
        
        self.state1_emission = list(self.state1_emission)    
    
    def set_parameters(self):
        """
        Create a transition matrix reflecting the following HMM:
            active site in [start, start + active_site_length]
            background sequence elsewhere
            
        Parameters:
            start: index at which the active site will begin
            active_site_length: length of the active site in generated data
            seq_length: total length of generated sequences
            state0_emission: array reflecting emission probability distribution (over amino acids) in state 0
            state1_emission: array reflecting emission probability distribution (over amino acids) in state 1
            background_emission: array reflecting emission probability distribution (over amino acids) in background
        
        Returns:
            M: A transition matrix of dimension (seq_length + active_site_length) x (seq_length + active_site_length)
                where the rows are grouped into the sections:
                    background
                    active site states along branch 0
                    active site states along branch 1
                    background
            startprob: array of initial probabilities (reflecting start at left of model)
            emissionprob: array of emission probabilities in each state
        """       
        start = self.start
        active_site_length = self.active_site_length
        seq_length = self.seq_length
        
        start_0_branch = start
        start_1_branch = start + active_site_length
        start_end_background = start + 2 * active_site_length
        n_components = seq_length + active_site_length
        
        M = np.zeros((n_components, n_components))
        
        state0_emission = list(self.state0_emission)
        state1_emission = list(self.state1_emission)
        background_emission = list(self.background_emission)
        
        emissionprob = []
        
        # Fill first background section
        for i in range(start_0_branch - 1):
            M[i, i+1] = 1
            emissionprob.append(background_emission)
        
        # Transition to one of two branches
        M[start_0_branch-1, start_0_branch] = 0.5
        M[start_0_branch-1, start_1_branch] = 0.5
        emissionprob.append(background_emission)
        
        # Fill in active site sections
        for i in range(start_0_branch, start_0_branch + active_site_length - 1):
            M[i, i+1] = 1
            emissionprob.append(state0_emission)
        
        M[start_0_branch + active_site_length - 1, start_end_background] = 1
        emissionprob.append(state0_emission)
        
        for i in range(start_0_branch + active_site_length, start_end_background):
            M[i, i+1] = 1
            emissionprob.append(state1_emission)
        
        # Fill in end background section
        for i in range(start_end_background, n_components-1):
            M[i, i+1] = 1
            emissionprob.append(background_emission)
        
        # Last component
        M[n_components - 1, n_components - 1] = 1
        emissionprob.append(background_emission)
        
        # Starting probabilities
        startprob = np.zeros((n_components,))
        startprob[0] = 1
        
        # Set class variables
        self.transmat = M
        self.startprob = startprob
        self.emissionprob = np.asarray(emissionprob)
        self.n_components = n_components


    
    
    
    
    
    
    
    
    
    
    