# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 19:07:27 2020

@author: NBOLLIG
"""

import pandas as pd
import matplotlib.pyplot as plt
from history import save_image
from pathlib import Path


def random_mut():
    """
    Analyze results from csv.
    """    
    
    mut_table = pd.read_csv("data/random_mut.csv")
    
    K = mut_table['k'].to_numpy()
    model_flip_rate = mut_table['model_flip_rate'].to_numpy()
    actual_flip_rate = mut_table['actual_flip_rate'].to_numpy()
    
    plt.plot(K, model_flip_rate, linestyle='-', color = 'k', label = "Model flip rate")
    plt.plot(K, actual_flip_rate, linestyle='--', color = 'k', label = "Actual label flip rate")
    plt.xlabel('Number of mutations')
    plt.title('Independent, uniformly distributed random mutations')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.ylim(0,0.5)
    save_image(plt, Path('data/'), "random_mut")
    plt.clf()