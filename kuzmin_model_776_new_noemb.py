# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 08:24:31 2021

"""
import numpy as np
import pickle

"""
# =============================================================================
Create Kuzmin dataset

This section is adapted from https://github.com/facebookresearch/esm

Kuzmin K, Adeniyi AE, DaSouza AK, Lim D, Nguyen H, Molina NR, et al. Machine learning methods 
accurately predict host specificity of coronaviruses based on spike sequences alone. 
Biochem Biophys Res Commun. 2020;533: 553–558. doi:10.1016/j.bbrc.2020.09.010

# =============================================================================
"""
from Bio import SeqIO # some BioPython that will come in handy

# Read the fasta-file and create a dictionary of its protein sequences

input_file_name = "Sequences.fasta"

# The fasta defline (name of a sequence) has the following format:
# Strain Name | Accession Number | Host Species | Virus Species) 
sequences_dictionary = {sequence.id : sequence.seq for sequence in SeqIO.parse(input_file_name,'fasta')}

# From the newly formed sequences_dictionary, we create 3 lists:
# a list of deflines,
# a list of sequences,
# and a list of target values

# We want to mark all sequences that belong to the viruses that can infect humans as 1 (i.e., target = 1), 
# all other sequences as 0 (i.e., target = 0)

human_virus_species_set =  {'Human_coronavirus_NL63', 'Betacoronavirus_1', 
                            'Human_coronavirus_HKU1', 'Severe_acute_respiratory_syndrome_related_coronavirus', 
                            'SARS_CoV_2', 'Human_coronavirus_229E', 'Middle_East_respiratory_syndrome_coronavirus'}

deflines = [entry for entry in sequences_dictionary.keys()]             # create a list of deflines
protein_sequences = [entry for entry in sequences_dictionary.values()]  # create a list of protein sequences 


# we assign 1 iff a virus species is one from the 7 human-infective

targets = [0]*len(deflines)
for i, defline in enumerate(deflines):
    for virus_species in human_virus_species_set:
        if virus_species in defline:
            targets[i] = 1

# We create a class fasta_sequence so that we would be able to use the sequence data easily 

class fasta_sequence:
    def __init__(self, defline, sequence, target, type_of_encoding = "onehot"):
        
        # we read the input data
        
        self.defline = defline
        self.sequence = sequence
        self.target = target
        
        # and create more descriptions of the input data
        
        # report the strain name (the 1st fiel of the defline)
        self.strain_name = defline.split("|")[0]
        # report the accession number (the 2nd fiel of the defline)
        self.accession_number = defline.split("|")[1]        
        # report the host species (the 3rd fiel of the defline)
        self.host_species = defline.split("|")[2]    
        # report the virus species (the 4th fiel of the defline)
        self.virus_species = defline.split("|")[3]
        
        
# We convert a string with the alphabet = 'ABCDEFGHIJKLMNPQRSTUVWXYZ-' 
# into either a list mapping chars to integers (called integer encoding),
# or a sparce list. In the latter, each amino acid is represented as an one-hot vector of length 25, 
# where each position, except one, is set to 0.  E.g., alanine is encoded as 10000000000000000000, 
# cystine is encoded as 01000000000000000000
# See the full table above.
# Symbol '-' is encoded as a zero-vector.

        def encoding(sequence, type_of_encoding):

            # define universe of possible input values
            alphabet = 'ABCDEFGHIJKLMNPQRSTUVWXYZ-'
            # define a mapping of chars to integers
            char_to_int = dict((c, i) for i, c in enumerate(alphabet))


            # integer encoding
            integer_encoded = [char_to_int[char] for char in sequence]

            # one-hot encoding
            onehot_encoded = list()
            for value in integer_encoded:
                letter = [0 for _ in range(len(alphabet)-1)]
                if value != len(alphabet)-1:
                    letter[value] = 1
                onehot_encoded.append(letter)
            flat_list = [item for sublist in onehot_encoded for item in sublist]

            if type_of_encoding == "onehot":
                return flat_list
            else:
                return integer_encoded
            
        #  we use the encoding function to create a new attribute for the sequence -- its encoding        
        self.encoded = encoding(sequence, type_of_encoding)

# we create a list of sequences as objects of the class fasta_sequence
# all sequences are encoded with one-hot encoding (it is the default option of the constructor of the class)
sequences = []
for i in range(0, len(deflines)):
    current_sequence = fasta_sequence(deflines[i],protein_sequences[i],targets[i])
    sequences.append(current_sequence)

# for a list of sequences, returns a list of encoded sequences and a list of targets

def EncodeAndTarget(list_of_sequences):
    # encoding the sequences
    list_of_encoded_sequences = [entry.encoded for entry in list_of_sequences]
    # creating lists of targets
    list_of_targets = [entry.target for entry in list_of_sequences]
    list_of_species = [entry.virus_species for entry in list_of_sequences]
    return list_of_encoded_sequences, list_of_targets, list_of_species

# =============================================================================
# Use all sequences for training/testing a model
# =============================================================================
training_sequences = sequences

# =============================================================================
# Get raw sequence encodings
# =============================================================================
N_POS = 2396
N_CHAR = 25

X, y, species = EncodeAndTarget(training_sequences)
X = np.array(X).reshape((-1, N_POS * N_CHAR))
y = np.array(y)
species = np.array(species)

# Randomize ordering
from sklearn.utils import shuffle
X, y, species, deflines, training_sequences = shuffle(X, y, species, deflines, training_sequences)

# =============================================================================
# Compute index sets
# =============================================================================
"""
We want to get index sets for all 7 human-infecting species. 
"""
human_virus_species_list = list(human_virus_species_set)

# Set up dictionary such that sp[index of species in human_virus_species_list]
# is a list of the indices in X and X_emb of that species. In addition, the value
# of sp['non-human'] is a list of indices for non-human-infecting viruses.
sp = {}
for i in range(len(human_virus_species_list)):
    sp[i] = []
sp['non-human'] = []

# Populate the dictionary
for i in range(len(species)):
    if species[i] in human_virus_species_list:
        species_idx = human_virus_species_list.index(species[i])
        sp[species_idx].append(i)
    else:
        sp['non-human'].append(i)


# Tests
n = 0
for i in range(len(human_virus_species_list)):
    n += len(sp[i])
assert(n == np.sum(y))
assert(len(sp['non-human']) == (len(y) - n))
assert(np.sum(y[sp['non-human']]) == 0)
for i in range(len(human_virus_species_list)):
    assert(np.all(y[sp[i]]==1)==True)

"""
We now have human_virus_species_list and sp dictionary as defined above.
"""

## =============================================================================
## Single Split Approach - For Testing
## =============================================================================

## Standard split as in Kuzmin paper
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#
#from keras.models import Sequential
#from keras.layers import LSTM
#from keras.layers import Dense
#from keras.layers import Bidirectional
#
#n = X_train.shape[0]
#X_train = X_train.reshape((n, N_POS, -1))
#num_features = X_train.shape[2]
#
#model = Sequential()
#model.add(Bidirectional(LSTM(64), input_shape = (N_POS, num_features)))
#model.add(Dense(16, activation='relu'))
#model.add(Dense(1, activation='sigmoid'))
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
## Train model
#model.fit(X_train, y_train, epochs=10, batch_size=64)
#
#_, acc = model.evaluate(X_test.reshape((X_test.shape[0], N_POS, -1)), y_test, verbose=0)
#print('Test Accuracy: %.2f' % (acc*100))
#
#from sklearn.metrics import roc_curve
#from sklearn.metrics import roc_auc_score
#import matplotlib.pyplot as plt
#
## ROC curve
#y_proba = model.predict(X_test.reshape((X_test.shape[0], N_POS, -1)))
#fpr, tpr, thresholds = roc_curve(y_test, y_proba)
#try:
#    auc = roc_auc_score(y_test, y_proba)
#except ValueError:
#    auc = 0
#    
#fig, ax = plt.subplots()
#ax.plot(fpr, tpr)
#ax.set(xlabel='False positive rate', ylabel='True positive rate', title=model_name + ' (AUC=%.3f)' % (auc,))
#ax.grid()
##fig.savefig(model_name + "_roc_curve.jpg", dpi=500)
#plt.show()
#
#
#
#
#
#
#from keras.models import Sequential
#from keras.layers import Conv1D
#from keras.layers import MaxPooling1D
#from keras.layers import Flatten
#from keras.layers import Dense
#
#
#n = X_train.shape[0]
#X_train = X_train.reshape((n, N_POS, -1))
#num_features = X_train.shape[2]
#
#model = Sequential()
#model.add(Conv1D(filters=32, kernel_size=5, input_shape=(N_POS, num_features)))
#model.add(MaxPooling1D(pool_size=5))
#model.add(Flatten())
#model.add(Dense(16, activation='relu'))
#model.add(Dense(1, activation='sigmoid'))
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
## Train model
#model.fit(X_train, y_train, epochs=10, batch_size=64)
#
#_, acc = model.evaluate(X_test.reshape((X_test.shape[0], N_POS, -1)), y_test, verbose=0)
#print('Test Accuracy: %.2f' % (acc*100))

# =============================================================================
# Define Kuzmin classifiers
# =============================================================================

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.base import clone


classifiers = {"SVM": SVC(probability = True, gamma = 'scale', random_state = 42), 
                 "Logistic Regression": LogisticRegression(C = 30.0, class_weight = 'balanced', 
                                                           solver = 'newton-cg', multi_class = 'multinomial', 
                                                           n_jobs = -1, random_state = 42),
                 "Decision Tree": DecisionTreeClassifier(random_state = 42),
                 "Random Forest": RandomForestClassifier(n_estimators = 500, max_leaf_nodes = 15, 
                                                         n_jobs= -1, random_state = 42),
                 "Baseline": DummyClassifier(strategy='constant', constant=1),
                 "LSTM": None,
                 "CNN": None}

# =============================================================================
# Define additional classifiers
# =============================================================================
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten

def make_LSTM(X_train, y_train, N_POS):   
    n = X_train.shape[0]
    X_train = X_train.reshape((n, N_POS, -1))
    num_features = X_train.shape[2]
    
    model = Sequential()
    model.add(Bidirectional(LSTM(64), input_shape = (N_POS, num_features)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def make_CNN(X_train, y_train, N_POS):
    n = X_train.shape[0]
    X_train = X_train.reshape((n, N_POS, -1))
    num_features = X_train.shape[2]
    
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=5, input_shape=(N_POS, num_features)))
    model.add(MaxPooling1D(pool_size=5))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

# =============================================================================
# Standard classification code
# =============================================================================
def classify(model_name, X_train, y_train, X_test, N_POS=2396):
    # Sklear classifiers
    if classifiers[model_name] != None:
        model = clone(classifiers[model_name])
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)
        assert(y_proba.shape[1] == 2)
        y_proba = y_proba[:,1]
        y_proba_train = model.predict_proba(X_train)
        y_proba_train = y_proba_train[:,1]
    
    # Keras classifiers
    if model_name == "LSTM" or model_name == "CNN":
        X_train = X_train.reshape((X_train.shape[0], N_POS, -1))
        X_test = X_test.reshape((X_test.shape[0], N_POS, -1))
        model = make_LSTM(X_train, y_train, N_POS=N_POS)
        model.fit(X_train, y_train, epochs=10, batch_size=64)
        y_proba = model.predict_proba(X_test)
        y_proba_train = model.predict_proba(X_train)        
    
    return y_proba, y_proba_train

# =============================================================================
# Standard evaluation code
# =============================================================================

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


def evaluate(y_proba, y_test, y_proba_train, y_train, model_name="", verbose=True):
    # PR curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
#    fig, ax = plt.subplots()
#    ax.plot(recall, precision)
#    ax.set(xlabel='Recall', ylabel='Precision', title=model_name + ' (AP=%.3f)' % (ap,))
#    ax.grid()
#    fig.savefig(model_name+"%i_pr_curve.jpg" % (int(time.time()),), dpi=500)
#    plt.show()
    
    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = 0
        
#    fig, ax = plt.subplots()
#    ax.plot(fpr, tpr)
#    ax.set(xlabel='False positive rate', ylabel='True positive rate', title=model_name + ' (AUC=%.3f)' % (auc,))
#    ax.grid()
#    #fig.savefig(model_name + "_roc_curve.jpg", dpi=500)
#    plt.show()
    
    # Evaluate on train set
    train_accuracy = accuracy_score(y_train, (y_proba_train >= 0.5).astype(int))
    train_recall = recall_score(y_train, (y_proba_train >= 0.5).astype(int))
    train_precision = precision_score(y_train, (y_proba_train >= 0.5).astype(int))
    train_f1 = f1_score(y_train, (y_proba_train >= 0.5).astype(int))
    
    # Evaluate on validation set
    test_accuracy = accuracy_score(y_test, (y_proba >= 0.5).astype(int))
    test_recall = recall_score(y_test, (y_proba >= 0.5).astype(int))
    test_precision = precision_score(y_test, (y_proba >= 0.5).astype(int))
    test_f1 = f1_score(y_test, (y_proba >= 0.5).astype(int))
    
    if verbose == True:
        print(model_name + ' Train Accuracy: %.2f' % (train_accuracy*100))
        print(model_name + ' Train Recall: %.2f' % (train_recall*100))
        print(model_name + ' Train Precision: %.2f' % (train_precision*100))
        print(model_name + ' Train F1: %.2f' % (train_f1*100))
        print(model_name + ' Test Accuracy: %.2f' % (test_accuracy*100))
        print(model_name + ' Test Recall: %.2f' % (test_recall*100))
        print(model_name + ' Test Precision: %.2f' % (test_precision*100))
        print(model_name + ' Test F1: %.2f' % (test_f1*100))
    
    return ap, auc, train_accuracy, train_recall, train_precision, train_f1, test_accuracy, test_recall, test_precision, test_f1
    
# =============================================================================
# Eval of LSTM PR curves
# =============================================================================
#temp_classifiers = {}
#temp_classifiers["LSTM"] = classifiers["LSTM"]
#temp_classifiers["Baseline"] = classifiers["Baseline"]
#
#classifiers = temp_classifiers

# Also uncomment PR curve in eval to see fold-specific curves
# Uncomment data for testing

# =============================================================================
# MAIN - Cross-validation with species-aware splitting
# =============================================================================
import pandas as pd
from sklearn.model_selection import GroupKFold

kfold = GroupKFold(n_splits=7)

Y_targets = []
output = []
i=0
Y_proba = {}
Y_proba_emb = {}
for model_name in classifiers:
    Y_proba[model_name] = []
    Y_proba_emb[model_name] = []


# Collect data for testing
X_TRAIN = []
X_TEST = []
Y_TRAIN = []
Y_TEST = []


for train, test in kfold.split(X[sp['non-human']], y[sp['non-human']], species[sp['non-human']]): # start by splitting only non-human data
    # Put the ith human-infecting virus species into the test set, the rest into train
    # Get indices of training species
    training_species = [k for k in [0,1,2,3,4,5,6] if k != i]
    training_species_idx = []
    for j in training_species:
        training_species_idx.extend(sp[j])
    
    # Create train and test arrays by concatenation
    X_train = np.vstack((X[sp['non-human']][train], X[training_species_idx]))
    X_test = np.vstack((X[sp['non-human']][test], X[sp[i]]))
    y_train = np.concatenate((y[sp['non-human']][train], y[training_species_idx]))
    y_test = np.concatenate((y[sp['non-human']][test], y[sp[i]]))
    
    # Shuffle arrays
    X_train, y_train = shuffle(X_train, y_train)
    X_test, y_test = shuffle(X_test, y_test)
    
    # Store data for testing
    X_TRAIN.append(X_train)
    X_TEST.append(X_test)
    Y_TRAIN.append(y_train)
    Y_TEST.append(y_test)
    
    print("*******************FOLD %i: %s*******************" % (i, human_virus_species_list[i]))
    print("Test size = %i" % (len(y_test),))
    print("Test non-human size = %i" % (len(X[sp['non-human']][test])),)
    print("Test human size = %i" % (len(X[sp[i]]),))
    print("Test pos class prevalence: %.3f" % (np.mean(y_test),))
    
    
    for model_name in classifiers:
        print("Training %s..." % (model_name,))
        
        # Raw sequence representation
        y_proba, y_proba_train = classify(model_name, X_train, y_train, X_test)
        results = evaluate(y_proba, y_test, y_proba_train, y_train, model_name)
        output.append((model_name, i, 'raw seq') + results)
        Y_proba[model_name].extend(y_proba)
    
    Y_targets.extend(y_test)
    i += 1

#"""
#For inspection of single-fold classification behavior. 
#
#1. Get train, test from kfold generator:
#    gen = kfold.split(X[sp['non-human']], y[sp['non-human']], species[sp['non-human']])
#    train,test = next(gen) # repeat for higher folds
#
#2. Do a single-fold run by running inner loop code above.
#
#3. Run the following after the single-fold run.
#
#    yp = list(y_proba.ravel())
#    yt = y_test
#    yp, yt = zip(*sorted(zip(yp, yt), reverse=True))
#
#Now we have ranked y_proba and parallel true labels in yt.
#
#Use the following function to determine species of instance at index i.
#"""
#def _get_sp(i):
#    i_y = list(y_proba.ravel()).index(yp[i]) # index of y_proba in y_test
#    x = X_test[i_y] # instance
#    i_x = np.where((X == x).all(axis=1))[0].item() # index of instance in X
#    return species[i_x]

print("*******************SUMMARY*******************")

output_df = pd.DataFrame(output, columns=['Model Name', 'Fold', 'Features', 'ap', 'auc', 'train_accuracy', 'train_recall', 'train_precision', 'train_f1', 'test_accuracy', 'test_recall', 'test_precision', 'test_f1'])
output_df.to_csv('cv_group_7_fold_noemb.csv')

# Pooled ROC curves
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

auc_baseline = roc_auc_score(Y_targets, np.ones(len(Y_targets)))

def get_ROC(Y_targets, Y_proba):
    fpr, tpr, _ = roc_curve(Y_targets, Y_proba)
    try:
        auc = roc_auc_score(Y_targets, Y_proba)
    except ValueError:
        auc = 0
    return fpr, tpr, auc

for key in Y_proba.keys(): # Loop over model types
    # Raw sequence
    fpr, tpr, auc = get_ROC(Y_targets, Y_proba[key])
    plt.step(fpr, tpr, where='post', label = key + ' (AUC=%.2f)' % (auc,) )
    
# Generate figure
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Sequence classification performance; baseline AUC=%.3f' % (auc_baseline,))
plt.legend(loc='upper left', fontsize=7, bbox_to_anchor=(1.05, 1))
plt.savefig("pooledROC_7_fold.jpg", dpi=400, bbox_inches = "tight")
plt.clf()

# Save data
data = (Y_proba, Y_targets)
pickle.dump(data, open( "data_group_7_fold_noemb.p", "wb" ))

## =============================================================================
## MAIN - CV with standard splitting
## =============================================================================
#print("Now doing standard splitting...")
#
#from sklearn.model_selection import KFold
#
#kfold = KFold(n_splits=7, shuffle=True)
#
#Y_targets = []
#output = []
#i=0
#Y_proba = {}
#Y_proba_emb = {}
#for model_name in classifiers:
#    Y_proba[model_name] = []
#    Y_proba_emb[model_name] = []
#
#
### Collect data for testing
##X_TRAIN = []
##X_TEST = []
##Y_TRAIN = []
##Y_TEST = []
#
#
#for train, test in kfold.split(X, y):
#    # Create train and test arrays by concatenation
#    X_train = X[train]
#    X_test = X[test]
#    X_emb_train = X_emb[train]
#    X_emb_test = X_emb[test]
#    y_train = y[train]
#    y_test = y[test]
#    
#    # Shuffle arrays
#    X_train, X_emb_train, y_train = shuffle(X_train, X_emb_train, y_train)
#    X_test, X_emb_test, y_test = shuffle(X_test, X_emb_test, y_test)
#    
##    # Store data for testing
##    X_TRAIN.append(X_train)
##    X_TEST.append(X_test)
##    Y_TRAIN.append(y_train)
##    Y_TEST.append(y_test)
#    
#    print("*******************FOLD %i: %s*******************" % (i, human_virus_species_list[i]))
#    print("Test size = %i" % (len(y_test),))
#    print("Test pos class prevalence: %.3f" % (np.mean(y_test),))
#    
#    
#    for model_name in classifiers:
#        # Raw sequence representation
#        y_proba, y_proba_train = classify(model_name, X_train, y_train, X_test)
#        results = evaluate(y_proba, y_test, y_proba_train, y_train, model_name)
#        output.append((model_name, i, 'raw seq') + results)
#        Y_proba[model_name].extend(y_proba)
#    
#        # Embedding representation
#        y_proba, y_proba_train = classify(model_name, X_emb_train, y_train, X_emb_test, N_POS=2560)
#        results = evaluate(y_proba, y_test, y_proba_train, y_train, model_name)
#        output.append((model_name, i, 'emb') + results)
#        Y_proba_emb[model_name].extend(y_proba)
#    
#    Y_targets.extend(y_test)
#    i += 1
#
#print("*******************SUMMARY*******************")
#
#output_df = pd.DataFrame(output, columns=['Model Name', 'Fold', 'Features', 'ap', 'auc', 'train_accuracy', 'train_recall', 'train_precision', 'train_f1', 'test_accuracy', 'test_recall', 'test_precision', 'test_f1'])
#output_df.to_csv('cv_bad_split_7_fold.csv')
#
## Pooled PR curves
#ap_baseline = average_precision_score(Y_targets, np.ones(len(Y_targets)))
#
#def get_PR(Y_targets, Y_proba):
#    precision, recall, _ = precision_recall_curve(Y_targets, Y_proba)
#    ap = average_precision_score(Y_targets, Y_proba)
#    return precision, recall, ap
#
#
#for key in Y_proba.keys(): # Loop over model types
#    # Raw sequence
#    precision, recall, ap = get_PR(Y_targets, Y_proba[key])
#    plt.step(recall, precision, where='post', label = key + ' - Raw Seq (AP=%.2f)' % (ap,) )
#    # Embedding
#    precision, recall, ap = get_PR(Y_targets, Y_proba_emb[key])
#    plt.step(recall, precision, where='post', label = key + ' - Emb (AP=%.2f)' % (ap,) )
#    
## Generate figure
#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.title('Sequence classification performance; baseline AP=%.3f)' % (ap_baseline,))
#plt.legend(loc='upper left', fontsize=7, bbox_to_anchor=(1.05, 1))
##plt.savefig("pooledPR_bad_split_7_fold.jpg", dpi=400, bbox_inches = "tight")
#plt.clf()
#
## Save data
#data = (Y_proba, Y_proba_emb, Y_targets)
#pickle.dump(data, open( "data_bad_split_7_fold.p", "wb" ))

## =============================================================================
## Feature Importance from LR
## =============================================================================
#
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#
#model = LogisticRegression(C = 30.0, class_weight = 'balanced', solver = 'newton-cg', multi_class = 'multinomial', n_jobs = -1, random_state = 42)
#model.fit(X,y)
#
#importance = np.array(model.coef_[0])
#
## Compute composite score at each position in multiple alignment
#importance = np.split(importance, N_POS) # split into arrays of values for each position
#composite_imps = []
#for array in importance:
#    composite_imps.append(np.linalg.norm(array)**2)
#composite_imps = np.array(composite_imps)
#composite_imps = composite_imps / np.sum(composite_imps)
#    
## List top 10 positions
#temp = np.argsort(-composite_imps)
#for t in temp[:10]:
#    print("Position %i importance: %.3f" % (t, composite_imps[t]))
#
## Draw plot
#plt.bar([i for i in range(len(composite_imps))], composite_imps, edgecolor='black', color='green')
##plt.savefig("LR_features.jpg", dpi=400, bbox_inches = "tight")
#plt.show()
#
## Investigate positions in human sequence
#for entry in sequences:
#    if entry.accession_number == 'QIQ49832':
#        seq = str(entry.sequence)
#
#for t in temp[:10]:
#    try:
#        _, pos = reformat_seq(seq, t)
#    except IndexError:
#        pos = 'Not present'
#    pos = str(pos)
#    print("Position %s importance: %.3f" % (pos, composite_imps[t]))

"""    
Compare to benchling

291, 327, 359: all near the start of the recepter binding region

"""

## =============================================================================
## Extra figures
## =============================================================================
## Simple split
#file = open('data_bad_split_7_fold.p', 'rb')
#data = pickle.load(file)
#file.close()
#
#Y_proba, Y_proba_emb, Y_targets = data
#
#ap_baseline = average_precision_score(Y_targets, np.ones(len(Y_targets)))
#
#def get_PR(Y_targets, Y_proba):
#    precision, recall, _ = precision_recall_curve(Y_targets, Y_proba)
#    ap = average_precision_score(Y_targets, Y_proba)
#    return precision, recall, ap
#
#for key in Y_proba.keys(): # Loop over model types
#    if key == "Baseline":
#        continue
#    # Raw sequence
#    precision, recall, ap = get_PR(Y_targets, Y_proba[key])
#    plt.step(recall, precision, where='post', label = key + ' (AP=%.2f)' % (ap,) )
#    
#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.title('Sequence classification performance; baseline AP=%.3f)' % (ap_baseline,))
#plt.legend(loc='upper left', fontsize=7, bbox_to_anchor=(1.05, 1))
#plt.savefig("bad_split_seq_PRs.jpg", dpi=400, bbox_inches = "tight")
#plt.clf()
#
## Species-aware split
#file = open('data_group_7_fold.p', 'rb')
#data = pickle.load(file)
#file.close()
#
#Y_proba, Y_proba_emb, Y_targets = data
#
#ap_baseline = average_precision_score(Y_targets, np.ones(len(Y_targets)))
#
#def get_PR(Y_targets, Y_proba):
#    precision, recall, _ = precision_recall_curve(Y_targets, Y_proba)
#    ap = average_precision_score(Y_targets, Y_proba)
#    return precision, recall, ap
#
#for key in Y_proba.keys(): # Loop over model types
#    if key == "Baseline":
#        continue
#    # Raw sequence
#    precision, recall, ap = get_PR(Y_targets, Y_proba[key])
#    plt.step(recall, precision, where='post', label = key + ' (AP=%.2f)' % (ap,) )
#    
#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.title('Sequence classification performance; baseline AP=%.3f)' % (ap_baseline,))
#plt.legend(loc='upper left', fontsize=7, bbox_to_anchor=(1.05, 1))
#plt.savefig("group_split_seq_PRs.jpg", dpi=400, bbox_inches = "tight")
#plt.clf()
#
## Species-aware split + Transformer embeddings
#file = open('data_group_7_fold.p', 'rb')
#data = pickle.load(file)
#file.close()
#
#Y_proba, Y_proba_emb, Y_targets = data
#
#ap_baseline = average_precision_score(Y_targets, np.ones(len(Y_targets)))
#
#def get_PR(Y_targets, Y_proba):
#    precision, recall, _ = precision_recall_curve(Y_targets, Y_proba)
#    ap = average_precision_score(Y_targets, Y_proba)
#    return precision, recall, ap
#
#for key in Y_proba.keys(): # Loop over model types
#    if key == "Baseline":
#        continue
#    # Embedding
#    precision, recall, ap = get_PR(Y_targets, Y_proba_emb[key])
#    plt.step(recall, precision, where='post', label = key + ' (AP=%.2f)' % (ap,) )
#    
## Generate figure
#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.title('Sequence classification performance; baseline AP=%.3f)' % (ap_baseline,))
#plt.legend(loc='upper left', fontsize=7, bbox_to_anchor=(1.05, 1))
#plt.savefig("group_split_emb_PRs.jpg", dpi=400, bbox_inches = "tight")
#plt.clf()
#
## Bare legend
#file = open('data_group_7_fold.p', 'rb')
#data = pickle.load(file)
#file.close()
#
#Y_proba, Y_proba_emb, Y_targets = data
#
#ap_baseline = average_precision_score(Y_targets, np.ones(len(Y_targets)))
#
#def get_PR(Y_targets, Y_proba):
#    precision, recall, _ = precision_recall_curve(Y_targets, Y_proba)
#    ap = average_precision_score(Y_targets, Y_proba)
#    return precision, recall, ap
#
#for key in Y_proba.keys(): # Loop over model types
#    if key == "Baseline":
#        continue
#    # Embedding
#    precision, recall, ap = get_PR(Y_targets, Y_proba_emb[key])
#    plt.step(recall, precision, where='post', label = key)
#    
## Generate figure
#plt.xlabel('Recall')
#plt.ylabel('Precision')
#plt.title('Sequence classification performance; baseline AP=%.3f)' % (ap_baseline,))
#plt.legend(loc='upper left', fontsize=7, bbox_to_anchor=(1.05, 1))
#plt.savefig("bare_legend.jpg", dpi=400, bbox_inches = "tight")
#plt.clf()

