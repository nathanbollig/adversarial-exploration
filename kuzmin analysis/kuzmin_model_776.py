# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 08:24:31 2021

@author: NBOLLIG
"""

# =============================================================================
# Create Kuzmin dataset
# =============================================================================
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
import numpy as np

training_sequences = sequences


# =============================================================================
# Analysis to get split point for embedding representation of sequence fragments
# =============================================================================

"""
from anotation of N25|ABD75545|Human|Human_coronavirus_HKU1 on Benchling, we
see that the Coronavirus S2 region starts at index 1518 of the multiple alignment.
We will convert each sequence in the multiple alignment back to an unaligned
sequence by removing gaps, and for each sequence, store the index of the start
of the S2 region.

Then we split each sequence at the appropriate index and create a fasta file
with each original sequence split into two sequences. The deflines have an additional
field indicating whether the sequence is up to (and not including) the S2 domain
or is the S2 domain only. 

This file is saved as "split_seqs.faa".
"""

#def reformat_seq(string, i=1518):
#    """
#    Takes a string and an index i. Returns the string without gap character '-'
#    and the index in the resulting string corresponding to the index i in the
#    original string.
#    """
#    
#    # Advance index to first non-gap character
#    while string[i] == '-':
#        i += 1
#    
#    # Count gap characters before i
#    offset = string[:i].count('-')
#    
#    # Remove gap chars
#    string = string.replace('-','')
#    
#    return string, i-offset
#
#from Bio.Seq import Seq
#from Bio.SeqRecord import SeqRecord
#
#records = []
#for seq in sequences:
#    seq_string = str(seq.sequence)
#    seq_string, i = reformat_seq(seq_string)
#    seq1 = seq_string[:i]
#    seq2 = seq_string[i:]
#    defline = seq.defline
#    rec1 = SeqRecord(Seq(seq1), id=defline+"|without S2")
#    rec2 = SeqRecord(Seq(seq2), id=defline+"|S2 only")
#    records.append(rec1)
#    records.append(rec2)
#
#SeqIO.write(records, "split_seqs.faa", "fasta")

# =============================================================================
# Get data for raw sequence data
# =============================================================================
N_POS = 2396
N_CHAR = 25

X, y, species = EncodeAndTarget(training_sequences)
X = np.array(X).reshape((-1, N_POS * N_CHAR))
y = np.array(y)
species = np.array(species)

# =============================================================================
# Get data for embeddings from file (produced externally)
# =============================================================================
import pickle
with open('embeddings.pkl', 'rb') as f:
    data = pickle.load(f)

# Goal is to create a dictionary of embeddings where keys are the first 4 fields of the defline
# and value is list [embedding array, label]
embeddings = {}

for header, emb in data.items():
    parsed_header = tuple(header.split('|'))

    # Get key from header
    key = parsed_header[:-1]
    
    # Get label
    if parsed_header[3].replace('-','_') in human_virus_species_set:
        label = 1
    else:
        label = 0    
    
    # Boolean flag for whether the embedding is for chunk 1 (or chunk 2) of sequence
    if parsed_header[-1].startswith('without S2'):
        part1 = True
    else:
        part1 = False

    # Load embedding into embeddings dictionary based on key
    if key in embeddings:
        embeddings[key][0] = np.concatenate((embeddings[key][0], emb))
    else:
        embeddings[key] = [emb, label]

# Transform to lists
X_emb = []
y_emb = []
species_emb = []
deflines = []

for key,val in embeddings.items():
    x = val[0]
    label = val[1]
    sp = key[3]
    X_emb.append(x)
    y_emb.append(label)
    species_emb.append(sp) 
    deflines.append(key)

X_emb = np.array(X_emb).reshape((-1, 2560))
y_emb = np.array(y_emb)

# =============================================================================
# Align data in X and X_emb
# =============================================================================

# We have data in X,y and in X_emb,y_emb. We will sort X_emb and y_emb so that
# the data is in the same order as X,y

def match_deflines(defline, emb_defline):
    """
    Return true if the deflines match. Handles formatting differences that emerged during this workflow.

        defline: defline from the original fasta_sequence object
        emb_defline: defline tuple extracted from the embedding file names
    """
    d = defline.split('|')
    e = emb_defline
    
    for i in range(4):
        if d[i].replace('/', '-').replace('_', '-') != e[i]:
            return False
    
    return True

assert(X.shape[0] == X_emb.shape[0])
N = X.shape[0]

for i in range(N):
    found = False
    X_defline = training_sequences[i].defline
    for j in range(i,N):
        if match_deflines(X_defline, deflines[j]) == True:
            found = True
            # cache items at i
            x = X_emb[i]
            label = y_emb[i]
            sp = species_emb[i]
            d = deflines[i]
            # replace at i with items at j
            X_emb[i] = X_emb[j]
            y_emb[i] = y_emb[j]
            species_emb[i] = species_emb[j]
            deflines[i] = deflines[j]
            # replace j with cached items from i
            X_emb[j] = x
            y_emb[j] = label
            species_emb[j] = sp
            deflines[j] = d
    assert(found == True)
    

# Tests
assert(np.all(y == y_emb))

for i in range(len(species)):
    if species[i].replace('/', '-').replace('_', '-') != species_emb[i]:
        assert(1==0)

"""
We now have the following aligned arrays:
    X - flattened one-hot sequence encodings
    X_emb - transformer embeddings
    y - labels
    species - species infected
    deflines
    training_sequences
    
"""

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

# Standard split as in Kuzmin paper
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

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
                 "ZeroR": DummyClassifier(strategy='constant', constant=1),
                 "LSTM": None}

# =============================================================================
# Define additional classifiers
# =============================================================================
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional

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

# =============================================================================
# Standard classificaiton code
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
    
    # LSTM
    if model_name == "LSTM":
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
#    #fig.savefig(model_name+"_pr_curve.jpg", dpi=500)
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
    
## =============================================================================
## MAIN - Train and evaluate models
## =============================================================================
#
#for model_name in classifiers:
#    y_proba, y_proba_train = classify(model_name, X_train, y_train, X_test)
#    evaluate(y_proba, y_test, y_proba_train, y_train, model_name)

# =============================================================================
# MAIN - Cross-validation with species-aware splitting
# =============================================================================
import pandas as pd
from sklearn.utils import shuffle
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


## Collect data for testing
#X_TRAIN = []
#X_TEST = []
#Y_TRAIN = []
#Y_TEST = []


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
    X_emb_train = np.vstack((X_emb[sp['non-human']][train], X_emb[training_species_idx]))
    X_emb_test = np.vstack((X_emb[sp['non-human']][test], X_emb[sp[i]]))
    y_train = np.concatenate((y[sp['non-human']][train], y[training_species_idx]))
    y_test = np.concatenate((y[sp['non-human']][test], y[sp[i]]))
    
    # Shuffle arrays
    X_train, X_emb_train, y_train = shuffle(X_train, X_emb_train, y_train)
    X_test, X_emb_test, y_test = shuffle(X_test, X_emb_test, y_test)
    
#    # Store data for testing
#    X_TRAIN.append(X_train)
#    X_TEST.append(X_test)
#    Y_TRAIN.append(y_train)
#    Y_TEST.append(y_test)
    
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
    
        # Embedding representation
        y_proba, y_proba_train = classify(model_name, X_emb_train, y_train, X_emb_test, N_POS=2560)
        results = evaluate(y_proba, y_test, y_proba_train, y_train, model_name)
        output.append((model_name, i, 'emb') + results)
        Y_proba_emb[model_name].extend(y_proba)
    
    Y_targets.extend(y_test)
    i += 1

print("*******************SUMMARY*******************")

output_df = pd.DataFrame(output, columns=['Model Name', 'Fold', 'Features', 'ap', 'auc', 'train_accuracy', 'train_recall', 'train_precision', 'train_f1', 'test_accuracy', 'test_recall', 'test_precision', 'test_f1'])
output_df.to_csv('cv_group_7_fold.csv')

# Pooled PR curves
ap_baseline = average_precision_score(Y_targets, np.ones(len(Y_targets)))

def get_PR(Y_targets, Y_proba):
    precision, recall, _ = precision_recall_curve(Y_targets, Y_proba)
    ap = average_precision_score(Y_targets, Y_proba)
    return precision, recall, ap


for key in Y_proba.keys(): # Loop over model types
    # Raw sequence
    precision, recall, ap = get_PR(Y_targets, Y_proba[key])
    plt.step(recall, precision, where='post', label = key + ' - Raw Seq (AP=%.2f)' % (ap,) )
    # Embedding
    precision, recall, ap = get_PR(Y_targets, Y_proba_emb[key])
    plt.step(recall, precision, where='post', label = key + ' - Emb (AP=%.2f)' % (ap,) )
    
# Generate figure
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Sequence classification performance; baseline AP=%.3f)' % (ap_baseline,))
plt.legend(loc='upper left', fontsize=7, bbox_to_anchor=(1.05, 1))
plt.savefig("pooledPR_7_fold.jpg", dpi=400)
plt.clf()

# =============================================================================
# MAIN - CV with standard splitting
# =============================================================================
print("Now doing standard splitting...")

from sklearn.model_selection import KFold

kfold = KFold(n_splits=7)

Y_targets = []
output = []
i=0
Y_proba = {}
Y_proba_emb = {}
for model_name in classifiers:
    Y_proba[model_name] = []
    Y_proba_emb[model_name] = []


## Collect data for testing
#X_TRAIN = []
#X_TEST = []
#Y_TRAIN = []
#Y_TEST = []


for train, test in kfold.split(X, y, species): # start by splitting only non-human data    
    # Create train and test arrays by concatenation
    X_train = X[train]
    X_test = X[test]
    X_emb_train = X_emb[train]
    X_emb_test = X_emb[test]
    y_train = y[train]
    y_test = y[test]
    
    # Shuffle arrays
    X_train, X_emb_train, y_train = shuffle(X_train, X_emb_train, y_train)
    X_test, X_emb_test, y_test = shuffle(X_test, X_emb_test, y_test)
    
#    # Store data for testing
#    X_TRAIN.append(X_train)
#    X_TEST.append(X_test)
#    Y_TRAIN.append(y_train)
#    Y_TEST.append(y_test)
    
    print("*******************FOLD %i: %s*******************" % (i, human_virus_species_list[i]))
    print("Test size = %i" % (len(y_test),))
    print("Test pos class prevalence: %.3f" % (np.mean(y_test),))
    
    
    for model_name in classifiers:
        # Raw sequence representation
        y_proba, y_proba_train = classify(model_name, X_train, y_train, X_test)
        results = evaluate(y_proba, y_test, y_proba_train, y_train, model_name)
        output.append((model_name, i, 'raw seq') + results)
        Y_proba[model_name].extend(y_proba)
    
        # Embedding representation
        y_proba, y_proba_train = classify(model_name, X_emb_train, y_train, X_emb_test, N_POS=2560)
        results = evaluate(y_proba, y_test, y_proba_train, y_train, model_name)
        output.append((model_name, i, 'emb') + results)
        Y_proba_emb[model_name].extend(y_proba)
    
    Y_targets.extend(y_test)
    i += 1

print("*******************SUMMARY*******************")

output_df = pd.DataFrame(output, columns=['Model Name', 'Fold', 'Features', 'ap', 'auc', 'train_accuracy', 'train_recall', 'train_precision', 'train_f1', 'test_accuracy', 'test_recall', 'test_precision', 'test_f1'])
output_df.to_csv('cv_bad_split_7_fold.csv')

# Pooled PR curves
ap_baseline = average_precision_score(Y_targets, np.ones(len(Y_targets)))

def get_PR(Y_targets, Y_proba):
    precision, recall, _ = precision_recall_curve(Y_targets, Y_proba)
    ap = average_precision_score(Y_targets, Y_proba)
    return precision, recall, ap


for key in Y_proba.keys(): # Loop over model types
    # Raw sequence
    precision, recall, ap = get_PR(Y_targets, Y_proba[key])
    plt.step(recall, precision, where='post', label = key + ' - Raw Seq (AP=%.2f)' % (ap,) )
    # Embedding
    precision, recall, ap = get_PR(Y_targets, Y_proba_emb[key])
    plt.step(recall, precision, where='post', label = key + ' - Emb (AP=%.2f)' % (ap,) )
    
# Generate figure
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Sequence classification performance; baseline AP=%.3f)' % (ap_baseline,))
plt.legend(loc='upper left', fontsize=7, bbox_to_anchor=(1.05, 1))
plt.savefig("pooledPR_bad_split_7_fold.jpg", dpi=400)
