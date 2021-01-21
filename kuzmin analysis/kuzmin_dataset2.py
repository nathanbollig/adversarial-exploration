# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 09:33:22 2020

Derived from https://github.com/kuzminkg/CoVs-S-pr/blob/master/CoVs-S-pr.ipynb

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
    return list_of_encoded_sequences, list_of_targets

# =============================================================================
# Withold SARS CoV 2 sequences
# =============================================================================

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

X_train, y_train = EncodeAndTarget(training_sequences)
X_SARS2, _ = EncodeAndTarget(SARS2_sequences)
x_bat, _ = EncodeAndTarget([bat_sequence])
x_bat = x_bat[0]

# =============================================================================
# KL Divergence visualization
# =============================================================================
# Create parallel lists of distributions
#    pos_distributions: at each position, a list of amino acids in that position within human sequences
#    neg_distributions: at each position, a list of amino acids in that position within non-human sequences

from utils import decode_from_one_hot
import collections
import numpy as np

N_POS = 2396
N_CHAR = 25
pos_distributions = [[] for _ in range(N_POS)]
neg_distributions = [[] for _ in range(N_POS)]

# Collect instances of amino acid occurrences at each position
for j in range(len(X_train)):
    x_encoded = X_train[j]
    x = decode_from_one_hot(x_encoded, n_positions=N_POS, n_characters=N_CHAR)
    if y_train[j] == 1:
        distributions = pos_distributions
    else:
        distributions = neg_distributions
    
    for i in range(N_POS):
        character = x[i]
        if character != -1:
            distributions[i].append(character)

# Form true distributions and compute KL divergence of human from non-human
KL = []
for i in range(N_POS):
    # Get distributions
    pos = collections.Counter(pos_distributions[i])
    total = sum(pos.values(), 0.0)
    for key in pos:
        pos[key] /= total
    
    neg = collections.Counter(neg_distributions[i])
    total = sum(neg.values(), 0.0)
    for key in neg:
        neg[key] /= total
    
    # Do not compute KL divergence if one or both classes have all gap characters
    if len(pos) == 0 or len(neg) == 0:
        KL.append(np.nan)
        continue
    
    # Compute KL divergence
    KL_partial = 0.0
    for j in range(N_CHAR):
        p = pos[j]
        q = neg[j]
        
        if p==0 and q==0:
            continue
        
        if p == 0:
            KL_partial += 0.0
        elif q != 0 and KL_partial != np.inf:
            KL_partial += p * np.log(p/q)
        else:
            KL_partial = np.inf
    
    KL.append(KL_partial)

# Replace inf with large number for visualization
KL = np.array(KL)
np.ma.masked_invalid(KL).max() # 5.866
KL[KL == np.inf] = 20

# Visualization of KL divergence with inf at 20
import matplotlib.pyplot as plt

plt.scatter(list(range(N_POS)), KL, facecolors='none', edgecolors='b', linewidth=0.5, s=2)
plt.xlabel('Position')
plt.ylabel('KL divergence')
plt.title('KL divergence, human from non-human')
plt.savefig('train_KL.jpg', dpi=400)
plt.clf()

# Histogram of inf
inf_indices = np.where(KL == 20)
bins = np.linspace(0, 2396, 35)
plt.hist(inf_indices, bins, density=False, facecolor='g', edgecolor='k')
plt.xlabel('Position')
plt.ylabel('Count')
plt.title('Positions where a residue in humans does not occur in non-humans')
plt.savefig('train_inf_hist.jpg', dpi=400)
plt.clf()


## =============================================================================
## Train a classifier
## =============================================================================
#from sklearn.model_selection import train_test_split
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import Flatten
#import numpy as np
#
## define the  model
#EPOCHS = 8
#N_POS = 2396
#N_CHAR = 25
#model = Sequential()
#model.add(Flatten(input_shape=(N_POS, N_CHAR)))
#model.add(Dense(1, activation='sigmoid', input_shape=(N_POS * N_CHAR,)))
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
## Split dataset
#X_train, X_test, y_train, y_test = train_test_split(np.array(X_train).reshape((-1, N_POS, N_CHAR)), np.array(y_train), test_size=0.2, random_state=1)
#
## Train model
#model.fit(X_train, y_train, epochs=EPOCHS, batch_size=64, verbose=1)
#
## Evaluate on train set
#result = {}
#_, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
#result['model_train_accuracy'] = train_accuracy
#print('Train Accuracy: %.2f' % (train_accuracy*100))
#
## Evaluate on validation set
#_, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
#result['model_test_accuracy'] = test_accuracy
#print('Test Accuracy: %.2f' % (test_accuracy*100))
#
## Fit on entire dataset for MM - so now train on test portion
#model.fit(X_test, y_test, epochs=2, batch_size=64, verbose=1)
#
## =============================================================================
## Apply MM to x_bat
## =============================================================================
#from perturbations import greedy_flip
#from history import History
#
#from pathlib import Path
#import pandas as pd
#aa_vocab = list('ABCDEFGHIJKLMNPQRSTUVWXYZ')
#
#x_perturb, data = greedy_flip(x_bat, 0, aa_vocab, model, generator=None, n_positions=N_POS, n_characters=N_CHAR, confidence_threshold = 0.99995)
#data = pd.DataFrame(data)
#
#h = History()
#h.set_dir(Path('data/'))
#h.result = result
#h.instance_summary = data
#h.save_tables()
#h.save()        
#    
#
## =============================================================================
## Compare against actual mutations from bat to human SARS CoV 2
## =============================================================================
#from utils import encode_as_one_hot
#"""
#From https://www-pnas-org.ezproxy.library.wisc.edu/content/117/17/9241
#
#Looking at evolution of node B from node A, involving two main mutations:
#    synonymous mutation T8782C - not captured in MM
#    C28144T changing a leucine (J, 9 or L, 11) to a serine (S, 17)
#
#Type C is characterized by nonsynonymous mutation G26144T which changes a glycine (G, 6) to a valine (V, 20)
#"""
#
#for entry in SARS2_sequences:
#    print(entry.defline)
#    
#X_SARS2, _ = EncodeAndTarget(SARS2_sequences)
#X_SARS2 = np.array(X_SARS2).reshape((-1, N_POS, N_CHAR))
#
#x_bat_enc = encode_as_one_hot(x_bat, n_positions=N_POS, n_characters=N_CHAR)
#x_perturb = encode_as_one_hot(x_perturb, n_positions=N_POS, n_characters=N_CHAR)
#
#"""
#Look at total distance of bat and MM from human i for each i
#"""
#j=0
#print("Differences of human instance from bat -> from perturbed bat SARS CoV 2")
#for x in X_SARS2:
#    initial_counter = 0
#    final_counter = 0
#    for i in range(N_POS):
#        if np.all(x[i] == x_bat_enc[i]) == False:
#            initial_counter += 1
#        if np.all(x[i] == x_perturb[i]) == False:
#            final_counter += 1
#    print("Human SARS-CoV-2 instance %i: %i -> %i \t Difference: %i" % (j, initial_counter, final_counter, final_counter - initial_counter))
#    j += 1
#
#SARS2_sequences[22].defline
#SARS2_sequences[32].defline
#
#"""
#The following shows the divergence of MM and human i from bat for each i
#This shows that there is a lot of similar mutations for human i from bat across i
#"""
#def positional_divergence(INDEX):
#    x = X_SARS2[INDEX]
#    MM_pos = []
#    human_pos = []
#    for i in range(N_POS):
#        if np.all(x[i] == x_bat_enc[i]) == False:
#            human_pos.append(i)
#        if np.all(x_bat_enc[i] == x_perturb[i]) == False:
#            MM_pos.append(i)
#    
#    import matplotlib.pyplot as plt
#    
#    bins = np.linspace(0, 2396, 100)
#    plt.hist(human_pos, bins, density=False, facecolor='b', edgecolor='k', alpha=0.75, label='Human %i' % (INDEX,))
#    plt.hist(MM_pos, bins, density=False, facecolor='g', edgecolor='k', alpha=0.4, label='MM')
#    
#    plt.xlabel('Position')
#    plt.ylabel('Count')
#    plt.legend(fontsize='small')
#    plt.title('Mutations in bat SARS CoV2 instance')
#    plt.show()
#
#for i in range(len(X_SARS2)):
#    positional_divergence(i)
#
#"""
#Find consensus mutational positions of human SARS CoV 2 from bat.
#"""
#from collections import defaultdict
#consensus = set()
#occurences = defaultdict(int)
#x0 = X_SARS2[0]
#for i in range(N_POS):
#    if np.all(x_bat_enc[i] == x[i]) == False:
#        consensus.add(i)
#        occurences[i] += 1
#        
#for j in range(1, len(X_SARS2)):
#    x = X_SARS2[j]
#    for i in list(consensus):
#        if np.all(x_bat_enc[i] == x[i]) != False:
#            consensus.remove(i)
#        else:
#            occurences[i] += 1
#
#import matplotlib.pyplot as plt
#
#MM_pos = []
#for i in range(N_POS):
#    if np.all(x_bat_enc[i] == x_perturb[i]) == False:
#        MM_pos.append(i)
#bins = np.linspace(0, 2396, 35)
#plt.hist(list(consensus), bins, density=False, facecolor='b', edgecolor='k', alpha=0.7, label='Contained in all human CoV 2')
#plt.hist(MM_pos, bins, density=False, facecolor='y', edgecolor='k', alpha=0.7, label='MM')
#
#plt.xlabel('Position')
#plt.ylabel('Count')
#plt.legend(fontsize='small')
#plt.title('Mutations relative to bat SARS CoV2')
#
#from history import save_image
#save_image(plt, h.dir_name, "MM in bat CoV2")
#plt.clf()
#
#"""
#All human SARS CoV 2 in this dataset have mutations 
# 1171,
# 1176,
# 1177,
# 1179,
# 1183,
# 1186,
# 1187,
# 1191,
# 1198,
#(as from consensus above). 
#
#MM produces mutations
# 1163,
# 1164,
# 1169,
# 1173,
# 1174,
# 1175,
# 1176,
# 1180,
#(as from MM_pos above).
#
#MM predicts a consensus mutation at pos 1176 (15 > 8), loss 0.04706
#AA: Q -> I
#Not a maximal loss
#
#
#"""
#from utils import decode_from_one_hot
#x_bat_integers = decode_from_one_hot(x_bat, n_positions=N_POS, n_characters=N_CHAR)
#search_string = "".join([aa_vocab[i] for i in x_bat_integers[1170:1185]]) # search for this in benchling AA alignment: SKPCNGQTGLNCYYP
#
#"""
#The pos 1176 mutation aligns with the spike rec binding region.
#
#Save history object with corrected instance_summary (gaps removed)
#"""
#
#def remove_gaps(x, data):
#    """
#    Take integer representation of x and pandas dataframe data like instance_summary.
#    Removes gaps from x and corrects pos_to_change in data to reflect positions in corrected sequence.
#    """
#    x = np.array(x).copy()
#    for index, row in data.iterrows():
#        position = int(row['pos_to_change'])
#        unique, counts = np.unique(x[:position], return_counts=True)
#        num_gaps = int(dict(zip(unique, counts)).get(-1))
#        new_position = position - num_gaps
#        data.loc[index, 'pos_to_change'] = new_position    
#    x = x[x != -1]    
#    return x, data
#
#x_perturb_integers = decode_from_one_hot(x_perturb, n_positions=N_POS, n_characters=N_CHAR)
#x_corrected, data_corrected = remove_gaps(x_perturb_integers, data)
#data_corrected = pd.DataFrame(data_corrected)
#
#h = History()
#h.set_dir(Path('data/'))
#h.result = result
#h.instance_summary = data_corrected
#h.save_tables()
#h.save()        
#
#"""
#Loss vs. positions - takes a long time and is too crowded
#Shows outliers throughout first 2/3 of positions
#"""
#
##length = len(x_corrected)
##losses = [[] for _ in range(length)]
##
##for _, row in data_corrected.iterrows():
##    losses[int(row['pos_to_change'])].append(row['max_loss_increase'])
##
### Plot histogram
##flierprops = dict(marker='o', markersize=0.8, markeredgecolor='g')
##plt.boxplot(losses, positions=list(range(length)), flierprops=flierprops)
##plt.xlabel('Position')
##plt.ylabel('Loss increase')
##plt.xticks(fontsize=6, rotation=90)
##plt.title('Loss increases achieved at each position')
##save_image(plt, h.dir_name, "loss_positional")
##plt.clf()
#
## =============================================================================
## Try including all SARS CoV 2 viruses in the training set
## =============================================================================
#
#X, y = EncodeAndTarget(sequences)
#
## define the  model
#EPOCHS = 8
#N_POS = 2396
#N_CHAR = 25
#model = Sequential()
#model.add(Flatten(input_shape=(N_POS, N_CHAR)))
#model.add(Dense(1, activation='sigmoid', input_shape=(N_POS * N_CHAR,)))
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
## Split dataset
#X_train, X_test, y_train, y_test = train_test_split(np.array(X).reshape((-1, N_POS, N_CHAR)), np.array(y), test_size=0.2, random_state=1)
#
## Train model
#model.fit(X_train, y_train, epochs=EPOCHS, batch_size=64, verbose=1)
#
## Evaluate on train set
#result = {}
#_, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
#result['model_train_accuracy'] = train_accuracy
#print('Train Accuracy: %.2f' % (train_accuracy*100))
#
## Evaluate on validation set
#_, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
#result['model_test_accuracy'] = test_accuracy
#print('Test Accuracy: %.2f' % (test_accuracy*100))
#
## Fit on entire dataset for MM - so now train on test portion
#model.fit(X_test, y_test, epochs=2, batch_size=64, verbose=1)
#
#x_perturb, data = greedy_flip(x_bat, 0, aa_vocab, model, generator=None, n_positions=N_POS, n_characters=N_CHAR, confidence_threshold = 0.9999)
#data = pd.DataFrame(data)
#
#x_bat_enc = encode_as_one_hot(x_bat, n_positions=N_POS, n_characters=N_CHAR)
#x_perturb = encode_as_one_hot(x_perturb, n_positions=N_POS, n_characters=N_CHAR)
#MM_pos = []
#for i in range(N_POS):
#    if np.all(x_bat_enc[i] == x_perturb[i]) == False:
#        MM_pos.append(i)
#bins = np.linspace(0, 2396, 35)
#plt.hist(list(consensus), bins, density=False, facecolor='b', edgecolor='k', alpha=0.7, label='Contained in all human CoV 2')
#plt.hist(MM_pos, bins, density=False, facecolor='y', edgecolor='k', alpha=0.7, label='MM')
#plt.xlabel('Position')
#plt.ylabel('Count')
#plt.legend(fontsize='small')
#plt.title('Mutations relative to bat SARS CoV2')
#
#h = History()
#h.set_dir(Path('data/'))
#h.result = result
#h.instance_summary = data
#h.save_tables()
#h.save()
#
#"""
#Bottom line is that even including the human SARS CoV 2 in the model's training 
#set does not improve the MM outcomes starting from the bat sequence
#"""     