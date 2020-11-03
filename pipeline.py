# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 09:40:53 2020

@author: NBOLLIG
"""
from seq_model import big_bang
import random
from sklearn.metrics import accuracy_score
import numpy as np
from keras.utils import to_categorical

def perturbation_pipeline(p = 0.5, n_generated = 5000, num_to_perturb = 500, perturb = None):
    model, X_list, y_list, generator = big_bang(num_instances=n_generated, p=p)
    _, _, X_test = X_list
    _, _, y_test = y_list  
    
    X_sample, y_sample = zip(*random.sample(list(zip(X_test, y_test)), num_to_perturb))
    
    # Convert list of one-hot encodings to list of sequences
    X = []
    for i in range(len(X_sample)):
        X.append(np.argmax(X_sample[i], axis=1, out=None).reshape(1, -1).tolist()[0])
    
    Y_initial = [] # initial labels
    Y_perturb = [] # actual labels of perturbed inputs
    Y_model = [] # model predicted labels of perturbed inputs
    
    for i in range(len(X)):
        x = X[i]
        y = y_sample[i]
        if (model.predict(to_categorical(x, num_classes=20).reshape(1,60,20)).item() > 0.5) == y:
            x_perturb = perturb(x)
            Y_perturb.append(generator.predict(x_perturb))
            Y_model.append(model.predict(to_categorical(x_perturb, num_classes=20).reshape((1,60,20))).item() > 0.5)
            Y_initial.append(y)
            
    model_flip_rate = 1 - accuracy_score(Y_initial, Y_model)
    actual_flip_rate = 1 - accuracy_score(Y_initial, Y_perturb)

    print("Model flip rate: %.5f" % (model_flip_rate,))
    print("Actual flip rate: %.5f" % (actual_flip_rate,))
    print("Number perturbations: %i" % (len(Y_initial),))

if __name__ == "__main__":
    from perturbations import no_perturb
    
    perturbation_pipeline(p=0.5, n_generated = 5000, num_to_perturb = 500, perturb = no_perturb)