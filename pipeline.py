# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 09:40:53 2020

@author: NBOLLIG
"""
from seq_model import big_bang
import random
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from keras.utils import to_categorical

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
    """
    Y_initial = [] # initial labels
    Y_perturb = [] # actual labels of perturbed inputs
    Y_model = [] # model predicted labels of perturbed inputs
    instance_data = []
    
    for i in range(len(X)):
        print("Perturbing %i..." % (i,))
        x = X[i]
        y = y_init[i]
        if perturb_args == None:
            x_perturb, data = perturb(x, y, aa_vocab, model)
        else:
            x_perturb, data = perturb(x, y, aa_vocab, model, **perturb_args)
        if type(data) == list:
            for d in data:
                d['instance'] = i + 1
        Y_perturb.append(generator.predict(x_perturb))
        Y_model.append(model.predict(to_categorical(x_perturb, num_classes=20).reshape((1,60,20))).item() > 0.5)
        Y_initial.append(y)
        instance_data.extend(data)
            
    instance_data = pd.DataFrame(instance_data)
    
    model_flip_rate = 1 - accuracy_score(Y_initial, Y_model)
    actual_flip_rate = 1 - accuracy_score(Y_initial, Y_perturb)
    
    if perturb_args == None:
        result = {}
    else:
        result = perturb_args.copy()

    result['model_flip_rate'] = model_flip_rate
    result['actual_flip_rate'] = actual_flip_rate
    result['avg mutations'] = instance_data.shape[0] / len(X)

    print("Model flip rate: %.5f" % (model_flip_rate,))
    print("Actual flip rate: %.5f" % (actual_flip_rate,))
    
    return result, instance_data

def perturbation_pipeline(p = 0.5, class_signal=10, n_generated = 5000, n_epochs = 75, num_to_perturb = 500, perturb = None, perturb_args = None):
    """
    Runs the perturbation pipeline with multiple sets of parameters to the perturbation method.
    
    perturb_args is a dictionary where keys are parameters to perturb, and values are parallel 
    lists corresponding to values of those parameters to run. For example, in the following case
    
        perturb_args[par1] = [val1, val2, val3]
        perturb_args[par2] = [val1, val2, val3]
    
    we would run the pipeline three times with values of par1 and par2 as indicated.
    """
    
    model, result, X_list, y_list, generator, aa_vocab = big_bang(class_signal=class_signal, num_instances=n_generated, p=p, n_epochs = n_epochs)
    _, _, X_test = X_list
    _, _, y_test = y_list
    
    X_sample, y_sample = zip(*random.sample(list(zip(X_test, y_test)), num_to_perturb))
    
    # Convert list of one-hot encodings to list of sequences
    X = []
    for i in range(len(X_sample)):
        X.append(np.argmax(X_sample[i], axis=1, out=None).reshape(1, -1).tolist()[0])
    
    # Filter to use only true positives and true negatives
    X_filtered = []
    y_filtered = []
    for i in range(len(X)):
        x = X[i]
        y = y_sample[i]
        if (model.predict(to_categorical(x, num_classes=20).reshape(1,60,20)).item() > 0.5) == y:
            X_filtered.append(x)
            y_filtered.append(y)
    
    n_perturbations = len(y_filtered)
    result['number_perturbations'] = n_perturbations
    
    # Cache experimental context
    result['p'] = p
    result['class_signal'] = class_signal
    result['n_generated'] = n_generated
    result['num_to_perturb'] = num_to_perturb
    result['perturb_method'] = perturb.__name__
    
    # Add model data to output
    output = pd.DataFrame(result, index = [0])
    instance_data_list = []
    
    if perturb_args == None:
        # Run perturb once
        rows, instance_data = perturb_one_set(model, generator, X_filtered, y_filtered, aa_vocab, perturb)
        rows = pd.DataFrame(rows, index = [1])
        instance_data_list = [instance_data]
    
    else:
        # Run the perturbation for the range of provided parameters in perturb_args
        first_param_list = next(iter(perturb_args.values()))
        rows = []
        for i in range(len(first_param_list)):
            args = {}
            
            for key,val in perturb_args.items():
                args[key] = val[i]
            
            result, instance_data = perturb_one_set(model, generator, X_filtered, y_filtered, aa_vocab, perturb, perturb_args = args)
            
            rows.append(pd.DataFrame(result, index = [i+1]))
            instance_data_list.append(instance_data)
    
        rows = pd.concat(rows, sort=False)
    
    # Organize output
    output = output.append(rows, sort=False).fillna('')  
    
    # Organize instance data
    instance_output = []
    for perturb_set_idx in range(len(instance_data_list)):
        instance_data = instance_data_list[perturb_set_idx]
        instance_data['perturb_set_idx'] = perturb_set_idx + 1
        instance_output.append(instance_data)
    
    instance_output = pd.concat(instance_output, sort=False, ignore_index=True)

    return output, instance_output

if __name__ == "__main__":
    from perturbations import no_perturb
    
    perturbation_pipeline(p=0.5, n_generated = 5000, num_to_perturb = 500, perturb = no_perturb)