# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 09:32:57 2020

@author: NBOLLIG
"""
import numpy as np
import random

class MarkovChain(object):
    def __init__(self, transition_matrix, states):
        """
        Initialize the MarkovChain instance.
 
        Parameters
        ----------
        transition_matrix: 2-D array
            A 2-D array representing the probabilities of change of 
            state in the Markov Chain.
 
        states: 1-D array 
            An array representing the states of the Markov Chain. It
            needs to be in the same order as transition_matrix.
        """
        self.transition_matrix = np.atleast_2d(transition_matrix)
        self.states = states
        self.index_dict = {self.states[index]: index for index in 
                           range(len(self.states))}
        self.state_dict = {index: self.states[index] for index in
                           range(len(self.states))}
 
    def next_state(self, current_state):
        """
        Returns the state of the random variable at the next time 
        instance.
 
        Parameters
        ----------
        current_state: str
            The current state of the system.
        """
        return np.random.choice(
         self.states, 
         p=self.transition_matrix[self.index_dict[current_state], :]
        )
 
    def generate_states(self, current_state, no=10):
        """
        Generates the next states of the system.
 
        Parameters
        ----------
        current_state: str
            The state of the current random variable.
 
        no: int
            The number of future states to generate.
        """
        future_states = []
        for i in range(no):
            next_state = self.next_state(current_state)
            future_states.append(next_state)
            current_state = next_state
        return future_states

class MarkovChainGenerator():

    def __init__(self, num_instances = 500, p = 0.5, seed0 = 0, seed1 = 1):
        self.num_instances = num_instances
        self.p = p
        self.seed0 = seed0
        self.seed1 = seed1

    def generate(self):
        """
        Generates list of sequences X and list of labels y.
        
        Parameters:
            num_instances: the number of total instances to generate
            p: positive (1) class prevalance
            seed0: random seed for generating Markov transition matrix for class 0
            seed1: random seed for generating Markov transition matrix for class 1
        
        Returns:
            X - list of sequences of length num_instances
            y - list of labels (0 or 1)
    
        """
        num_instances = self.num_instances
        p = self.p
        seed0 = self.seed0
        seed1 = self.seed1
        
        aa_list = list('ARNDCEQGHILKMFPSTWYV')
        n = len(aa_list)
        num1 = int(num_instances*p)
        num0 = num_instances - num1
        
        # Set transition matrices
        np.random.seed(seed0)
        m0 = np.random.rand(n, n)
        for i in range(m0.shape[0]):
            m0[i,:] = m0[i,:] / np.sum(m0[i,:])
        
        np.random.seed(seed1)
        m1 = np.random.rand(n, n)
        m1 = m1 / np.sum(m1)
        for i in range(m1.shape[0]):
            m1[i,:] = m1[i,:] / np.sum(m1[i,:])
        
        # Create Markov Chain objects
        chain0 = MarkovChain(transition_matrix = m0, states = aa_list)
        chain1 = MarkovChain(transition_matrix = m1, states = aa_list)
        
        X0 = []
        for i in range(num0):
            x0 = chain0.generate_states(current_state = random.choice(aa_list), no=100)
            X0.append(''.join(x0))
        
        X1 = []
        for i in range(num1):
            x1 = chain1.generate_states(current_state = random.choice(aa_list), no=100)
            X1.append(''.join(x1))
        
        # Create X and y lists
        X = X0 + X1
        y = list(np.zeros(num0, dtype=int)) + list(np.ones(num1, dtype=int))
        
        # Shuffle data
        data = list(zip(X, y))
        random.shuffle(data)
        X, y = zip(*data)
        
        # Convert y to an array
        y = np.array(y, dtype=int)
        
        return X, y, aa_list, m0, m1
    
    
    
    
    
    
    