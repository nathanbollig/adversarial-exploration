# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 09:37:51 2020

@author: NBOLLIG
"""
from seq_model import big_bang
from HMM_generator import HMMGenerator
from pipeline import perturbation_pipeline

from pathlib import Path
import pandas as pd
import os
import datetime

def save_output(output, dir_name, exp_name):
    timestamp = str(int(datetime.datetime.now().timestamp()))
    path = os.path.join(dir_name, exp_name + timestamp +".csv")
    output.to_csv(path)

def exp1(dir_name):
    """
    We assign nonzero probabilities to all amino acids in positive class - do this by 
    averaging distribution derived from motif (which does not have all aa) with the state 0 
    emission probability distribution.

    This introduces a hyperparameter k into the generative model and the mixture is
    (motif_distribution * k + state0_distribution)/(k+1).
    
    This experiment trains a LSTM on data generated using a range of k values, and reports the model accuracy.
    
    GOAL: Show task difficulty correlates with the signal to noise ratio.
    """
    output = []
    
    for k in [0, 1, 2, 3, 4, 5, 10, 15, 20, 50, 100]:
        for n_epochs in [25, 75]:
            _, result, _, _, _, _ = big_bang(num_instances=2000, p=0.5, class_signal=k, n_epochs=n_epochs)
            train_acc = result['model_train_accuracy']
            val_acc = result['model_val_accuracy']
            row = [k, n_epochs, train_acc, val_acc]
            output.append(row)
    
    output = pd.DataFrame(output, columns = ['class_signal', 'n_epochs', 'model_train_accuracy', 'model_val_accuracy'])
    
    save_output(output, dir_name, "exp1")
    return output    

def exp2(dir_name):
    """
    Vary p and run above type of experiment with fixed k.
    
    GOAL: Show task difficulty correlates with the class prevalance.
    """
    output = []
    
    for p in [0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95]:
        _, result, _, _, _, _ = big_bang(num_instances=2000, p=p, class_signal=10, n_epochs=75)
        train_acc = result['model_train_accuracy']
        val_acc = result['model_val_accuracy']
        row = [p, train_acc, val_acc]
        output.append(row)
    
    output = pd.DataFrame(output, columns = ['p', 'model_train_accuracy', 'model_val_accuracy'])
    
    save_output(output, dir_name, "exp2")
    return output    

def exp3(dir_name):
    """
    Quantify error of HMM inference, regardless of model behavior.
    """
    gen = HMMGenerator()
    count = 0
    total = 10000
    
    
    for i in range(total):
        seq, y = gen.generate_one_sequence()
        y_inference = gen.predict(seq)
        if y != y_inference:
            count += 1
            
    error = count/total
    
    output = pd.DataFrame([[count, total, error]], columns = ['err_count', 'total', 'error'])
    
    print("Fraction of incorrect inferences: %.5f" % (error,))
    
    save_output(output, dir_name, "exp3")
    return output    

def exp4(dir_name):
    """
    Run perturbation pipeline with no perturbation. Illustrates error of HMM inference 
    on a set of true positives and true negatives.
    """
    from perturbations import no_perturb
    
    output = perturbation_pipeline(p=0.5, n_generated = 5000, num_to_perturb = 500, perturb = no_perturb, n_epochs = 75)
    save_output(output, dir_name, "exp4")
    return output

def exp5(dir_name):
    """
    Run perturbation pipeline with mutate 1 or k characters perturbation.
    
    Former way of organizing the pipeline code.
    """
    from perturbations import random_pt_mutations
    from functools import partial
    perturb = partial(random_pt_mutations, k=10)
    perturb.__name__ = 'random_pt_mutations'
    
    output = perturbation_pipeline(p=0.5, n_generated = 5000, num_to_perturb = 500, perturb = perturb, n_epochs = 25)
    save_output(output, dir_name, "exp5")
    return output

def exp6(dir_name):
    """
    Run perturbation pipeline with mutate 1 or k characters perturbation.
    """
    from perturbations import random_pt_mutations
    
    perturb_args = {}
    perturb_args['k'] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,25,30,35,40,50,60]
    
    output = perturbation_pipeline(p=0.5, n_generated = 5000, num_to_perturb = 500, perturb = random_pt_mutations, n_epochs = 25, perturb_args = perturb_args)
    save_output(output, dir_name, "exp6")
    return output

def exp7(dir_name):
    """
    Run single HotFlip.
    """
    from perturbations import hot_flip
    
    output = perturbation_pipeline(p=0.5, n_generated = 5000, num_to_perturb = 100, perturb = hot_flip, n_epochs = 25)
    save_output(output, dir_name, "exp7")
    return output

if __name__ == "__main__":
    dir_name = Path('data/')
    
    exp7(dir_name)
