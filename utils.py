# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 20:50:11 2020

@author: NBOLLIG
"""
import numpy as np
from keras.utils import to_categorical

def decode_from_one_hot(x):
    return np.argmax(x, axis=1, out=None).reshape(1, -1).tolist()[0]

def encode_as_one_hot(x):
    return to_categorical(x, num_classes=20).reshape(60,20)