# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 09:33:22 2020

Derived from https://github.com/kuzminkg/CoVs-S-pr/blob/master/CoVs-S-pr.ipynb

@author: NBOLLIG
"""
from Bio import SeqIO # some BioPython that will come in handy
import pandas as pd
import numpy as np
from utils import decode_from_one_hot
import collections
    
def prepare():
    # =============================================================================
    # Create Kuzmin dataset
    # =============================================================================
    
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
    MERS_human_sequences = []
    MERS_camel_sequences = []
    
    for entry in sequences:
        if entry.virus_species == "Middle_East_respiratory_syndrome_coronavirus" and entry.host_species == 'Human':
            MERS_human_sequences.append(entry)
        elif entry.virus_species == "Middle_East_respiratory_syndrome_coronavirus" and entry.host_species == 'Camel':
            MERS_camel_sequences.append(entry)
            training_sequences.append(entry) # If the camel sequence should be in the training set
        else:
            training_sequences.append(entry)
    
    X_train, y_train = EncodeAndTarget(training_sequences)
    X_MERS_human, _ = EncodeAndTarget(MERS_human_sequences)
    x_MERS_camel, _ = EncodeAndTarget(MERS_camel_sequences)
    
    return X_train, y_train, X_MERS_human, x_MERS_camel
    
def trial(camel_index, X_train, y_train, X_MERS_human, x_MERS_camel):
    """
    Run the analysis using the given index as the refence camel MERS sequence.
    """
    x_camel = x_MERS_camel[camel_index]
    
    # =============================================================================
    # Compute non-human and human vectors
    # =============================================================================
    # Create parallel lists of distributions
    #    pos_distributions: at each position, a list of amino acids in that position within human sequences
    #    neg_distributions: at each position, a list of amino acids in that position within non-human sequences
    
    
    N_POS = 2396
    N_CHAR = 25
    pos_aa = [[] for _ in range(N_POS)]
    neg_aa = [[] for _ in range(N_POS)]
    
    # Collect instances of amino acid occurrences at each position
    for j in range(len(X_train)):
        x_encoded = X_train[j]
        x = decode_from_one_hot(x_encoded, n_positions=N_POS, n_characters=N_CHAR)
        if y_train[j] == 1:
            aa = pos_aa
        else:
            aa = neg_aa
        
        for i in range(N_POS):
            character = x[i]
            if character != -1:
                aa[i].append(character)
    
    pos_dist = [[] for _ in range(N_POS)]
    neg_dist = [[] for _ in range(N_POS)]
    
    # Form true distributions
    for i in range(N_POS):
        # Get distributions
        pos = collections.Counter(pos_aa[i])
        total = sum(pos.values(), 0.0)
        for key in pos:
            pos[key] /= total
        
        neg = collections.Counter(neg_aa[i])
        total = sum(neg.values(), 0.0)
        for key in neg:
            neg[key] /= total
       
        if len(pos) == 0 or len(neg) == 0:
            continue
        
        pos_vector = [0] * N_CHAR
        for j in range(N_CHAR):
            if j in pos:
                pos_vector[j] = pos[j]
    
        neg_vector = [0] * N_CHAR
        for j in range(N_CHAR):
            if j in neg:
                neg_vector[j] = neg[j]
    
        pos_dist[i] = np.array(pos_vector)
        neg_dist[i] = np.array(neg_vector)
    
    
    # =============================================================================
    # Visualization a: Nonhuman vs. human (in training set)
    # =============================================================================
    
    from numpy.linalg import norm
    def cosine_similarity(a,b):
        if len(a) == 0 or len(b) == 0:
            return np.nan
        return np.inner(a, b) / (norm(a) * norm(b))
    
    cossim_vals = [cosine_similarity(pos_dist[i], neg_dist[i]) for i in range(N_POS)]
    
    # Visualization of KL divergence with inf at 20
    import matplotlib.pyplot as plt
    
    bad_indices = np.isnan(cossim_vals)
    good_indices = ~bad_indices
    x = np.array(range(N_POS))[good_indices]
    y = np.array(cossim_vals)[good_indices]
    sim = np.mean(y)
    
    plt.scatter(x, y, facecolors='none', edgecolors='b', linewidth=0.5, s=2)
    plt.xlabel('Position')
    plt.ylabel('Cosine similarity')
    plt.title('Similarity of human and non-human classes at each position: %.4f' % (sim,))
    #plt.savefig('kuz3_cossim_classes_scatter.jpg', dpi=400)
    plt.clf()
    
    plt.hist2d(x, y, (25,25), cmap=plt.cm.jet)
    plt.colorbar()
    plt.xlabel('Position')
    plt.ylabel('Cosine similarity')
    plt.title('Similarity of human and non-human classes at each position: %.4f' % (sim,))
    #plt.savefig('kuz3_cossim_classes_heatmap.jpg', dpi=400)
    plt.clf()
    
    a = sim
    
    # =============================================================================
    # Compute camel distribution
    # =============================================================================
    
    x = decode_from_one_hot(x_camel, n_positions=N_POS, n_characters=N_CHAR)
    
    x_camel_dist = [[] for _ in range(N_POS)]
    for i in range(N_POS):
        aa_index = x[i]
        if aa_index == -1:
            continue
        
        vec = np.zeros((N_CHAR,))
        vec[aa_index] = 1
        
        x_camel_dist[i] = vec
    
    # =============================================================================
    # Visualization b: camel MERS vs. human (in training set)
    # =============================================================================
    
    cossim_vals = [cosine_similarity(pos_dist[i], x_camel_dist[i]) for i in range(N_POS)]
    
    bad_indices = np.isnan(cossim_vals)
    good_indices = ~bad_indices
    x = np.array(range(N_POS))[good_indices]
    y = np.array(cossim_vals)[good_indices]
    sim = np.mean(y)
    
    plt.scatter(x, y, facecolors='none', edgecolors='b', linewidth=0.5, s=2)
    plt.xlabel('Position')
    plt.ylabel('Cosine similarity')
    plt.title('Similarity of camel MERS and human classes at each position: %.4f' % (sim,))
    #plt.savefig('kuz3_cossim_camel_scatter.jpg', dpi=400)
    plt.clf()
    
    plt.hist2d(x, y, (25,25), cmap=plt.cm.jet)
    plt.colorbar()
    plt.xlabel('Position')
    plt.ylabel('Cosine similarity')
    plt.title('Similarity of camel MERS and human classes at each position: %.4f' % (sim,))
    #plt.savefig('kuz3_cossim_camel_heatmap.jpg', dpi=400)
    plt.clf()
    
    cossim_b = cossim_vals
    b=sim
    
    # =============================================================================
    # Compute MM sequence
    # =============================================================================
    
    from sklearn.model_selection import train_test_split
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Flatten
    
    # define the  model
    EPOCHS = 8
    N_POS = 2396
    N_CHAR = 25
    model = Sequential()
    model.add(Flatten(input_shape=(N_POS, N_CHAR)))
    model.add(Dense(1, activation='sigmoid', input_shape=(N_POS * N_CHAR,)))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(np.array(X_train).reshape((-1, N_POS, N_CHAR)), np.array(y_train), test_size=0.2, random_state=1)
    
    # Train model
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=64, verbose=1)
    
    # Evaluate on train set
    result = {}
    _, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    result['model_train_accuracy'] = train_accuracy
    print('Train Accuracy: %.2f' % (train_accuracy*100))
    
    # Evaluate on validation set
    _, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    result['model_test_accuracy'] = test_accuracy
    print('Test Accuracy: %.2f' % (test_accuracy*100))
    
    # Fit on entire dataset for MM - so now train on test portion
    model.fit(X_test, y_test, epochs=2, batch_size=64, verbose=1)
    
    from perturbations import greedy_flip
    
    aa_vocab = list('ABCDEFGHIJKLMNPQRSTUVWXYZ')
    
    x_MM, _ = greedy_flip(x_camel, 0, aa_vocab, model, generator=None, n_positions=N_POS, n_characters=N_CHAR, confidence_threshold = 0.99995)
    
    # =============================================================================
    # Compute MM distribution
    # =============================================================================
    
    x = x_MM
    
    x_MM_dist = [[] for _ in range(N_POS)]
    for i in range(N_POS):
        aa_index = x[i]
        if aa_index == -1:
            continue
        
        vec = np.zeros((N_CHAR,))
        vec[aa_index] = 1
        
        x_MM_dist[i] = vec
    
    # =============================================================================
    # Visualization c: MM vs. human (in training set)
    # =============================================================================
    
    cossim_vals = [cosine_similarity(pos_dist[i], x_MM_dist[i]) for i in range(N_POS)]
    
    bad_indices = np.isnan(cossim_vals)
    good_indices = ~bad_indices
    x = np.array(range(N_POS))[good_indices]
    y = np.array(cossim_vals)[good_indices]
    sim = np.mean(y)
    
    plt.scatter(x, y, facecolors='none', edgecolors='b', linewidth=0.5, s=2)
    plt.xlabel('Position')
    plt.ylabel('Cosine similarity')
    plt.title('Similarity of MM and human classes at each position: %.4f' % (sim,))
    #plt.savefig('kuz3_cossim_MM_scatter.jpg', dpi=400)
    plt.clf()
    
    plt.hist2d(x, y, (25,25), cmap=plt.cm.jet)
    plt.colorbar()
    plt.xlabel('Position')
    plt.ylabel('Cosine similarity')
    plt.title('Similarity of MM and human classes at each position: %.4f' % (sim,))
    #plt.savefig('kuz3_cossim_MM_heatmap.jpg', dpi=400)
    plt.clf()
    
    cossim_c = cossim_vals
    c=sim
    
    # =============================================================================
    # Visualization d: camel vs. MM
    # =============================================================================
    
    cossim_vals = [cosine_similarity(x_camel_dist[i], x_MM_dist[i]) for i in range(N_POS)]
    
    bad_indices = np.isnan(cossim_vals)
    good_indices = ~bad_indices
    x = np.array(range(N_POS))[good_indices]
    y = np.array(cossim_vals)[good_indices]
    sim = np.mean(y)
    
    plt.scatter(x, y, facecolors='none', edgecolors='b', linewidth=0.5, s=2)
    plt.xlabel('Position')
    plt.ylabel('Cosine similarity')
    plt.title('Similarity of camel and MM at each position: %.4f' % (sim,))
    #plt.savefig('kuz3_cossim_camelvMM_scatter.jpg', dpi=400)
    plt.clf()
    
    plt.hist2d(x, y, (25,25), cmap=plt.cm.jet)
    plt.colorbar()
    plt.xlabel('Position')
    plt.ylabel('Cosine similarity')
    plt.title('Similarity of camel and MM at each position: %.4f' % (sim,))
    #plt.savefig('kuz3_cossim_camelvMM_heatmap.jpg', dpi=400)
    plt.clf()
    
    d=sim
    
    # =============================================================================
    # SARS CoV 2 distribution
    # =============================================================================
    aa = [[] for _ in range(N_POS)]
    
    # Collect instances of amino acid occurrences at each position
    for j in range(len(X_MERS_human)):
        x_encoded = X_MERS_human[j]
        x = decode_from_one_hot(x_encoded, n_positions=N_POS, n_characters=N_CHAR)
        
        for i in range(N_POS):
            character = x[i]
            if character != -1:
                aa[i].append(character)
    
    SARS_dist = [[] for _ in range(N_POS)]
    
    # Form true distributions
    for i in range(N_POS):
        # Get distributions
        dist = collections.Counter(aa[i])
        total = sum(dist.values(), 0.0)
        for key in dist:
            dist[key] /= total
       
        if len(dist) == 0:
            continue
        
        vector = [0] * N_CHAR
        for j in range(N_CHAR):
            if j in dist:
                vector[j] = dist[j]
    
        SARS_dist[i] = np.array(vector)
    
    # =============================================================================
    # Visualization e: human (training set) vs. MERS human
    # =============================================================================
    
    cossim_vals = [cosine_similarity(pos_dist[i], SARS_dist[i]) for i in range(N_POS)]
    
    bad_indices = np.isnan(cossim_vals)
    good_indices = ~bad_indices
    x = np.array(range(N_POS))[good_indices]
    y = np.array(cossim_vals)[good_indices]
    sim = np.mean(y)
    
    plt.scatter(x, y, facecolors='none', edgecolors='b', linewidth=0.5, s=2)
    plt.xlabel('Position')
    plt.ylabel('Cosine similarity')
    plt.title('Similarity of human non-MERS and MERS at each position: %.4f' % (sim,))
    #plt.savefig('kuz3_cossim_humanvMERS_scatter.jpg', dpi=400)
    plt.clf()
    
    plt.hist2d(x, y, (25,25), cmap=plt.cm.jet)
    plt.colorbar()
    plt.xlabel('Position')
    plt.ylabel('Cosine similarity')
    plt.title('Similarity of human non-MERS and MERS at each position: %.4f' % (sim,))
    #plt.savefig('kuz3_cossim_humanvMERS_heatmap.jpg', dpi=400)
    plt.clf()
    
    e=sim
    # =============================================================================
    # Visualization f: camel vs. human MERS
    # =============================================================================
    
    cossim_vals = [cosine_similarity(x_camel_dist[i], SARS_dist[i]) for i in range(N_POS)]
    
    bad_indices = np.isnan(cossim_vals)
    good_indices = ~bad_indices
    x = np.array(range(N_POS))[good_indices]
    y = np.array(cossim_vals)[good_indices]
    sim = np.mean(y)
    
    plt.scatter(x, y, facecolors='none', edgecolors='b', linewidth=0.5, s=2)
    plt.xlabel('Position')
    plt.ylabel('Cosine similarity')
    plt.title('Similarity of camel and human MERS at each position: %.4f' % (sim,))
    #plt.savefig('kuz3_cossim_camelvCoV2_scatter.jpg', dpi=400)
    plt.clf()
    
    plt.hist2d(x, y, (25,25), cmap=plt.cm.jet)
    plt.colorbar()
    plt.xlabel('Position')
    plt.ylabel('Cosine similarity')
    plt.title('Similarity of camel and human MERS at each position: %.4f' % (sim,))
    #plt.savefig('kuz3_cossim_camelvCoV2_heatmap.jpg', dpi=400)
    plt.clf()
    
    cossim_f = cossim_vals
    f=sim
    
    # =============================================================================
    # Visualization g: MM vs. human MERS
    # =============================================================================
    
    cossim_vals = [cosine_similarity(SARS_dist[i], x_MM_dist[i]) for i in range(N_POS)]
    
    bad_indices = np.isnan(cossim_vals)
    good_indices = ~bad_indices
    x = np.array(range(N_POS))[good_indices]
    y = np.array(cossim_vals)[good_indices]
    sim = np.mean(y)
    
    plt.scatter(x, y, facecolors='none', edgecolors='b', linewidth=0.5, s=2)
    plt.xlabel('Position')
    plt.ylabel('Cosine similarity')
    plt.title('Similarity of MM and human MERS at each position: %.4f' % (sim,))
    #plt.savefig('kuz3_cossim_MMvMERS_scatter.jpg', dpi=400)
    plt.clf()
    
    plt.hist2d(x, y, (25,25), cmap=plt.cm.jet)
    plt.colorbar()
    plt.xlabel('Position')
    plt.ylabel('Cosine similarity')
    plt.title('Similarity of MM and human MERS at each position: %.4f' % (sim,))
    #plt.savefig('kuz3_cossim_MMvMERS_heatmap.jpg', dpi=400)
    plt.clf()
    
    cossim_g = cossim_vals
    g=sim
    # =============================================================================
    # Comparing b (camel vs human) to c (MM vs. human)
    # =============================================================================
    
    # Note np.all(np.isnan(cossim_b) == np.isnan(cossim_c)) is true
    
    bad_indices = np.isnan(cossim_b)
    good_indices = ~bad_indices
    x = np.array(range(N_POS))[good_indices]
    bb = np.array(cossim_b)[good_indices]
    cc = np.array(cossim_c)[good_indices]
    y = cc-bb
    avg = np.mean(y) # 0.009397929011614386
    
    plt.scatter(x, y, facecolors='none', edgecolors='r', linewidth=0.5, s=2)
    plt.xlabel('Position')
    plt.ylabel('Difference in cosine similarity')
    plt.title('Comparing similarity of MM to human vs. camel to human (mean: %.3f)' % (avg,))
    plt.ylim(-1.1,1.1)
    #plt.savefig('kuz3_cossim_comp1_scatter.jpg', dpi=400)
    plt.clf()
    
    # =============================================================================
    # Comparing f (bat vs human SARS CoV 2) to g (MM vs. human MERS)
    # =============================================================================
    
    # Note np.all(np.isnan(cossim_f) == np.isnan(cossim_g)) is true
    
    bad_indices = np.isnan(cossim_f)
    good_indices = ~bad_indices
    x = np.array(range(N_POS))[good_indices]
    ff = np.array(cossim_f)[good_indices]
    gg = np.array(cossim_g)[good_indices]
    y = gg-ff
    avg = np.mean(y) # 0.009397929011614386
    
    plt.scatter(x, y, facecolors='none', edgecolors='r', linewidth=0.5, s=2)
    plt.xlabel('Position')
    plt.ylabel('Difference in cosine similarity')
    plt.title('Comparing similarity of MM to h MERS vs. bat to h MERS (mean: %.3f)' % (avg,))
    plt.ylim(-1.1,1.1)
    #plt.savefig('kuz3_cossim_comp2_scatter.jpg', dpi=400)
    plt.clf()

    return a, b, c, d, e, f, g

if __name__ == '__main__':    
    X_train, y_train, X_MERS_human, x_MERS_camel = prepare()
    
    #trial(0, X_train, y_train, X_MERS_human, x_MERS_camel) # single analysis
    
    # Multiple analysis
    data = []
    for i in range(len(x_MERS_camel)):
        vals = trial(i, X_train, y_train, X_MERS_human, x_MERS_camel)
        data.append(list(vals))
    data = pd.DataFrame(np.array(data), columns=['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    data.to_csv('data.csv', index=False)