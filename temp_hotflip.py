# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 10:22:45 2020

@author: NBOLLIG
"""

from seq_model import big_bang
import random
import numpy as np

from keras import backend as K

# =============================================================================
# Train model and RETAIN ONE HOT ENCODING
# =============================================================================
p = 0.5
class_signal=10
n_generated = 5000
n_epochs = 5
num_to_perturb = 500

model, result, X_list, y_list, generator, aa_vocab = big_bang(class_signal=class_signal, num_instances=n_generated, p=p, n_epochs = n_epochs)
_, _, X_test = X_list
_, _, y_test = y_list

X_sample, y_sample = zip(*random.sample(list(zip(X_test, y_test)), num_to_perturb))

# RETAIN list of one-hot encodings
X = X_sample

# Filter to use only true positives and true negatives
X_filtered = []
y_filtered = []
for i in range(len(X)):
    x = X[i]
    y = y_sample[i]
    if (model.predict(np.asarray(X[i]).reshape((1,60,20))).item() > 0.5) == y:
        X_filtered.append(x)
        y_filtered.append(y)

n_perturbations = len(y_filtered)

# =============================================================================
# Get gradient of loss at specific input wrt inputs
# =============================================================================

# Context: model, generator, X_filtered, y_filtered, aa_vocab

# Specific input x and its label y
x = X_filtered[0]
y = np.asarray(y_filtered).reshape((-1,1))[0]

from keras.layers import Input

# Set up function to compute gradient
y_true = Input(shape=(1,))
ce = K.binary_crossentropy(y_true, model.output)
grad_ce = K.gradients(ce, model.inputs)
func = K.function(model.inputs + [y_true], grad_ce)

# Gradient of loss at (x,y) with respect to inputs
output = func([np.asarray(x).reshape((1,60,20)), y])[0][0]

# =============================================================================
# Find single best flip
# =============================================================================

# Character sequence for x
a_vector = np.argmax(x, axis=1, out=None).reshape(1, -1).tolist()[0]

# Find character flip that causes maximum increase in loss
max_loss_increase = 0
pos_to_change = None
current_char_idx = None
new_char_idx = None

for i in range(60):
    a = a_vector[i]
    for b in range(20):
        loss_b = output[i][b]
        loss_a = output[i][a]
        loss_increase = loss_b - loss_a
        if loss_increase > max_loss_increase:
            max_loss_increase = loss_increase
            pos_to_change = i
            current_char_idx = a
            new_char_idx = b
    
# =============================================================================
# Next: Beam search for multiple flips
# =============================================================================










