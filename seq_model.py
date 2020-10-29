# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 10:08:01 2020

@author: NBOLLIG
"""
#from generator import MarkovChainGenerator
from HMM_generator import HMMGenerator
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import pickle


def encode_raw_sequences(X_raw, aa_vocab):
    """
    Parameters:
        X _raw: list of sequence strings
        aa_vocab: list of aa tokens
    
    Output:
        List of sequences each encoded as an array of size (100, 20) since every 
        sequence is of length 100 and there are 20 amino acids
    """
#    # Encode each instance as a list of indices into aa vocabulary [NEEDED FOR CHARACTER SEQ, NOT INDEX SEQ]
#    aa_dict = {}
#    for i, aa in enumerate(aa_vocab):
#        aa_dict[aa] = i
#    
#    X_seq = []
#    for x in X_raw:
#        x = list(x)
#        x = [aa_dict[aa] for aa in x]
#        X_seq.append(x)

    # Bypass character to index conversion above
    X_seq = X_raw

    # Transform to one-hot encoding
    X = to_categorical(X_seq)
    
    return X

def create_LSTM(X_train, X_val, y_train, y_val):
    # define the  model
    model = Sequential()
    model.add(LSTM(128, input_shape=(60,20)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Train model
    model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=1)
    
    # Evaluate on test set
    _, accuracy = model.evaluate(X_val, y_val, verbose=0)
    print('Accuracy: %.2f' % (accuracy*100))
    
    return model

def big_bang(num_instances=5000, p=0.5, seed0 = 0, seed1 = 1):
    """
    Generates sequence data and trains a model.
    
    Parameters:
        num_instances: the number of total instances to generate
        p: positive (1) class prevalance
    
    
    Returns:
        model: trained Keras model
        X: list [X_train, X_val, X_test]
        y: list [y_train, y_val, y_test]
    """
    # Generate data
    
#    mcg = MarkovChainGenerator(num_instances=num_instances, p=p, seed0=seed0, seed1=seed1)
#    X_raw, y, aa_vocab, m0, m1 = mcg.generate()
    
    gen = HMMGenerator()
    X_raw, y = gen.generate(n_samples=num_instances)
    aa_vocab = gen.aa_list
    
    # Encode X
    X = encode_raw_sequences(X_raw, aa_vocab)
    
    # Split into train, validation, test (80/10/10)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=1)
    
    # Define model
    model = create_LSTM(X_train, X_val, y_train, y_val)
    
    return model, [X_train, X_val, X_test], [y_train, y_val, y_test]

def main(name, num_instances=5000, p=0.5, seed0 = 0, seed1 = 1):
    """
    Parameters:
        name: a string identifier for the experiment
        others as in big_bang
    """
    
    model, X_list, y_list = big_bang(num_instances=num_instances, p=p, seed0 = seed0, seed1 = seed1)
    model.save(name+'_keras_model')
    with open(name+'_X_list.pkl', 'wb') as f:
        pickle.dump(X_list, f)
    with open(name+'_y_list.pkl', 'wb') as f:
        pickle.dump(y_list, f)

#    with open(name+'_m0.pkl', 'wb') as f:
#        pickle.dump(m0, f)
#    with open(name+'_m1.pkl', 'wb') as f:
#        pickle.dump(m1, f)

if __name__ == "__main__":
    main('test_HMM')
