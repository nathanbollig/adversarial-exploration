# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 09:40:53 2020

@author: NBOLLIG
"""
from seq_model import big_bang
from history import History
import random
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from pathlib import Path
import os
import gc

#def perturbation_pipeline(p = 0.5, class_signal=10, n_generated = 5000, n_epochs = 75, num_to_perturb = 500, perturb = None):
#    model, result, X_list, y_list, generator, aa_vocab = big_bang(class_signal=class_signal, num_instances=n_generated, p=p, n_epochs = n_epochs)
#    _, _, X_test = X_list
#    _, _, y_test = y_list  
#    
#    X_sample, y_sample = zip(*random.sample(list(zip(X_test, y_test)), num_to_perturb))
#    
#    # Convert list of one-hot encodings to list of sequences
#    X = []
#    for i in range(len(X_sample)):
#        X.append(np.argmax(X_sample[i], axis=1, out=None).reshape(1, -1).tolist()[0])
#    
#    Y_initial = [] # initial labels
#    Y_perturb = [] # actual labels of perturbed inputs
#    Y_model = [] # model predicted labels of perturbed inputs
#    
#    for i in range(len(X)):
#        x = X[i]
#        y = y_sample[i]
#        if (model.predict(to_categorical(x, num_classes=20).reshape(1,60,20)).item() > 0.5) == y:
#            x_perturb = perturb(x, aa_vocab)
#            Y_perturb.append(generator.predict(x_perturb))
#            Y_model.append(model.predict(to_categorical(x_perturb, num_classes=20).reshape((1,60,20))).item() > 0.5)
#            Y_initial.append(y)
#            
#    model_flip_rate = 1 - accuracy_score(Y_initial, Y_model)
#    actual_flip_rate = 1 - accuracy_score(Y_initial, Y_perturb)
#    n_perturbations = len(Y_initial)
#
#    result['model_flip_rate'] = model_flip_rate
#    result['actual_flip_rate'] = actual_flip_rate
#    result['number_perturbations'] = n_perturbations
#
#    print("Model flip rate: %.5f" % (model_flip_rate,))
#    print("Actual flip rate: %.5f" % (actual_flip_rate,))
#    print("Number perturbations: %i" % (n_perturbations,))
#    
#    # Cache experimental context
#    result['p'] = p
#    result['class_signal'] = class_signal
#    result['n_generated'] = n_generated
#    result['num_to_perturb'] = num_to_perturb
#    result['perturb_method'] = perturb.__name__
#    
#    return result

def perturb_one_set(model, generator, X, y_init, aa_vocab, perturb, perturb_args=None):
    """
    Do one iteration of perturbation on the sampled instaces (using one particular set of arguments to the perturbation method).
    
    perturb_args is a dictionary where keys are parameters to perturb and values are
    each a single value for the corresponding parameter.
    
    Returns:
        set_summary_line - a dictionary that will correspond to a single line in the set_summary
        instance_data - df of instance-level data for this perturbation set
    """
    Y_initial = [] # initial labels
    Y_perturb = [] # actual labels of perturbed inputs
    Y_model = [] # model predicted labels of perturbed inputs
    instance_data = []
    
    for i in range(len(X)):
        print("Perturbing item %i" % (i,), end='')
        
        x = X[i]
        y = y_init[i]
        if perturb_args == None:
            x_perturb, data = perturb(x, y, aa_vocab, model, generator) # data is a list of dictionaries
        else:
            x_perturb, data = perturb(x, y, aa_vocab, model, generator, **perturb_args)
        if type(data) == list:
            for d in data:
                d['instance'] = i + 1
        Y_perturb.append(generator.predict(x_perturb))
        Y_model.append(model.predict(to_categorical(x_perturb, num_classes=20).reshape((1,60,20))).item() > 0.5)
        Y_initial.append(y)
        instance_data.extend(data) # maintain list of dictionaries
                
    # Compute block of instance_summary
    instance_data = pd.DataFrame(instance_data) # convert list of dictionaries to df
    
    # Compute line for set summary
    model_flip_rate = 1 - accuracy_score(Y_initial, Y_model)
    actual_flip_rate = 1 - accuracy_score(Y_initial, Y_perturb)
    
    if perturb_args == None:
        set_summary_line = {}
    else:
        set_summary_line = perturb_args.copy()

    set_summary_line['model_flip_rate'] = model_flip_rate
    set_summary_line['actual_flip_rate'] = actual_flip_rate
    set_summary_line['avg mutations'] = instance_data.shape[0] / len(X)

    print("Model flip rate: %.5f" % (model_flip_rate,))
    print("Actual flip rate: %.5f" % (actual_flip_rate,))
    
    return set_summary_line, instance_data

def perturbation_pipeline(p = 0.5, class_signal=10, n_generated = 5000, model_type=None, n_epochs = 75, num_to_perturb = 500, perturb = None, perturb_args = None, dir_name="", legacy_output = False):
    """
    Runs the perturbation pipeline. If multiple sets of parameters are provided to the perturb method, 
    perturbation takes place in multiple independent runs.
    
    perturb_args is a dictionary where keys are parameters to perturb, and values are parallel 
    lists corresponding to values of those parameters to run. For example, in the following case
    
        perturb_args[par1] = [val1, val2, val3]
        perturb_args[par2] = [val1, val2, val3]
    
    we would run the pipeline three times with values of par1 and par2 as indicated.
    
    In this code, each call to perturbation_one_set is called a "perturbation set".
    
    When legacy_output is false, pipeline returns a history object with the following fields.
        result (dict) - pipeline-level (global) strings and scalars, including:
            model_train_accuracy
            model_val_accuracy
            number_perturbations
            p
            class_signal
            n_generated
            num_to_perturb
            perturb_method
        
        set_summary (df) - summary data, one line per perturbation set, with columns:
            perturb_set_idx | perturb params | model_flip_rate | actual_flip_rate | avg_mutations
        
        instance_summary (df) - instance-level data across perturbation sets, with columns:
            perturb_set_idx | instance | change_number | ...
        
    """
    h = History()
    if dir_name=="":
        dir_name = Path(os.getcwd())
    h.set_dir(dir_name)
    
    model, h.result, X_list, y_list, generator, aa_vocab = big_bang(class_signal=class_signal, num_instances=n_generated, p=p, model_type=model_type, n_epochs = n_epochs)
    _, _, X_test = X_list
    _, _, y_test = y_list
    
    # Convert list of one-hot encodings to list of sequences
    X = []
    for i in range(len(X_test)):
        X.append(np.argmax(X_test[i], axis=1, out=None).reshape(1, -1).tolist()[0])
    
    # Filter to use only true negatives
    X_filtered = []
    y_filtered = []
    for i in range(len(X)):
        x = X[i]
        y = y_test[i]
        if y==0 and model.predict(to_categorical(x, num_classes=20).reshape(1,60,20)).item() < 0.5:
            X_filtered.append(x)
            y_filtered.append(y)
    
    h.result['num_true_negs'] = len(y_filtered)
    
    # Restrict to num_to_perturb
    X_filtered, y_filtered = zip(*random.sample(list(zip(X_filtered, y_filtered)), num_to_perturb))
    
    # Cache experimental context
    h.result['p'] = p
    h.result['class_signal'] = class_signal
    h.result['n_generated'] = n_generated
    h.result['num_to_perturb'] = num_to_perturb
    h.result['perturb_method'] = perturb.__name__
    
    # Add model data to output for legacy compatibility
    if legacy_output == True:
        output = pd.DataFrame(h.result, index = [0])
    
    if perturb_args == None:
        # Run perturb once
        set_summary_line, instance_data = perturb_one_set(model, generator, X_filtered, y_filtered, aa_vocab, perturb)
        
        # Create set summary
        h.set_summary = pd.DataFrame(set_summary_line, index = [1])
        h.set_summary['perturb_set_idx'] = 1
        
        # Create list of (one) instance-level df
        h.instance_data_list = [instance_data]
    
    else:
        h.instance_data_list = []
        
        # Run the perturbation for the range of provided parameters in perturb_args
        first_param_list = next(iter(perturb_args.values()))
        h.set_summary_rows = []
        for i in range(len(first_param_list)):
            print("Running set %i..." % (i,))
            
            args = {}
            
            for key,val in perturb_args.items():
                args[key] = val[i]
            
            set_summary_line, instance_data = perturb_one_set(model, generator, X_filtered, y_filtered, aa_vocab, perturb, perturb_args = args)
            set_summary_line = pd.DataFrame(set_summary_line, index = [i+1])
            set_summary_line['perturb_set_idx'] = i+1
            h.set_summary_rows.append(pd.DataFrame(set_summary_line, index = [i+1])) # maintain list of single-row dfs
            
            # Create a list of dfs
            h.instance_data_list.append(instance_data)
            
            # Save current history
            h.save()
            gc.collect()
        
        # Create set summary
        # Combine all single-row dfs into one df
        h.set_summary = pd.concat(h.set_summary_rows)
        del h.set_summary_rows
    
    # Create instance summary
    # Currently instance_data_list is a list of dfs
    for perturb_set_idx in range(len(h.instance_data_list)):
        h.instance_data_list[perturb_set_idx]['perturb_set_idx'] = perturb_set_idx + 1
    
    h.instance_summary = pd.concat(h.instance_data_list, ignore_index=True)
    del h.instance_data_list

    # Output results
    if legacy_output == True:
        output = output.append(h.set_summary, sort=False).fillna('')  # add to results in 0th row
        return output, h.instance_summary
    
    gc.collect()
    return h

if __name__ == "__main__":
    from perturbations import no_perturb
    
    perturbation_pipeline(p=0.5, n_generated = 5000, num_to_perturb = 500, perturb = no_perturb)