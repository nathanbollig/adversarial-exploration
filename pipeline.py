# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 09:40:53 2020

@author: NBOLLIG
"""
from seq_model import big_bang
import random
from sklearn.metrics import accuracy_score

def perturbation_pipeline(n_generated = 5000, num_to_perturb = 500, perturb = None):
    model, X_list, y_list, generator = big_bang(num_instances=n_generated)
    _, _, X_test = X_list
    _, _, y_test = y_list
    
    X_sample, y_sample = zip(*random.sample(list(zip(X_test, y_test)), num_to_perturb))
    
    Y_initial = [] # initial labels
    Y_perturb = [] # actual labels of perturbed inputs
    Y_model = [] # model predicted labels of perturbed inputs
    
    for i in range(len(X_sample)):
        x = X_sample[i]
        y = y_sample[i]
        if (model.predict(x.reshape(1,60,20)).item() > 0.5) == y:
            x_perturb = perturb(x)
            Y_perturb.append(generator.predict(x_perturb))
            Y_model.append(model.predict(x_perturb.reshape((1,60,20))).item() > 0.5)
            Y_initial.append(y)
            
    model_flip_rate = 1 - accuracy_score(Y_initial, Y_model)
    actual_flip_rate = 1 - accuracy_score(Y_initial, Y_perturb)

    print("Model flip rate: %.2f" % (model_flip_rate,))
    print("Actual flip rate: %.2f" % (actual_flip_rate,))

if __name__ == "__main__":
    from perturbations import no_perturb
    
    perturbation_pipeline(n_generated = 5000, num_to_perturb = 500, perturb = no_perturb)