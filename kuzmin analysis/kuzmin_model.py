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
# Withold SARS CoV 2 sequences
# =============================================================================
import numpy as np

training_sequences = []
SARS2_sequences = []

for entry in sequences:
    if entry.virus_species == 'SARS_CoV_2' and entry.host_species == 'Human':
        SARS2_sequences.append(entry)
    elif entry.virus_species == 'SARS_CoV_2':
        bat_sequence = entry
        training_sequences.append(entry) # If the bat sequence should be in the training set
    else:
        training_sequences.append(entry)

# =============================================================================
# Form and split data set
# =============================================================================
N_POS = 2396
N_CHAR = 25


X, y, species = EncodeAndTarget(training_sequences)
X = np.array(X).reshape((-1, N_POS, N_CHAR))
y = np.array(y)

from sklearn.model_selection import GroupKFold
group_kfold = GroupKFold(n_splits=2)

for train_index, test_index in group_kfold.split(X, y, species):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# =============================================================================
# Train model
# =============================================================================

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
import numpy as np

# define the  model
EPOCHS = 8
N_POS = 2396
N_CHAR = 25
model = Sequential()
model.add(Flatten(input_shape=(N_POS, N_CHAR)))
model.add(Dense(1, activation='sigmoid', input_shape=(N_POS * N_CHAR,)))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=64, verbose=1)

# =============================================================================
# Evaluate model
# =============================================================================

# PR curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

y_proba = model.predict_proba(X_test)
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)
fig, ax = plt.subplots()
ax.plot(recall, precision)
ax.set(xlabel='Recall', ylabel='Precision', title='Sequence classification model (AP=%.3f)' % (ap,))
ax.grid()
fig.savefig("pr_curve.jpg", dpi=500)
plt.show()

# ROC curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

fpr, tpr, thresholds = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)
fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.set(xlabel='False positive rate', ylabel='True positive rate', title='Sequence classification model (AUC=%.3f)' % (auc,))
ax.grid()
fig.savefig("roc_curve.jpg", dpi=500)
plt.show()

# Evaluate on train set
result = {}
_, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
result['model_train_accuracy'] = train_accuracy
print('Train Accuracy: %.2f' % (train_accuracy*100))

# Evaluate on validation set
_, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
result['model_test_accuracy'] = test_accuracy
print('Test Accuracy: %.2f' % (test_accuracy*100))