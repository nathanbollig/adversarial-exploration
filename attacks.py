# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 07:56:18 2020

@author: NBOLLIG
"""
import numpy as np

"""
CleverHans: https://github.com/tensorflow/cleverhans
FoolBox: https://github.com/bethgelab/foolbox
adversarial-robustness-toolbox: 
    https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/notebooks/README.md
    https://adversarial-robustness-toolbox.readthedocs.io/en/latest/
    ()
"""



#def main():
#    from model import get_IMDB_dataset
##    from sklearn.svm import SVC
#    
#    from keras.models import Sequential
#    from keras import layers
#    from keras.optimizers import Adam
#    
#    from art.estimators.classification import SklearnClassifier
#    from art.attacks.evasion import CarliniLInfMethod
#
#    X, y, X_holdout, y_holdout = get_IMDB_dataset()
#    
#    X_train = X[:20000]
#    y_train = y[:20000]
#    X_val = X[20000:]
#    y_val = y[20000:]
#
## =============================================================================
##   Create Model
## =============================================================================
#
#    embedding_dim = 50
#    vocab_size = 20000
#    
#    model = Sequential()
#    model.add(layers.Embedding(input_dim=vocab_size, 
#                               output_dim=embedding_dim, 
#                               input_length=100))
#    model.add(layers.Flatten())
#    
#    model.add(layers.Dense(10, activation='relu'))
#    model.add(layers.Dense(1, activation='sigmoid'))
#    optimizer = Adam(lr=0.001)
#    model.compile(optimizer=optimizer,
#                  loss='binary_crossentropy',
#                  metrics=['accuracy'])
#    model.summary()
#    
#    history = model.fit(X_train, y_train,
#                    epochs=10,
#                    verbose=True,
#                    validation_data=(X_val, y_val),
#                    batch_size=10)
#
## =============================================================================
## Clever Hans
## =============================================================================
#
#    from cleverhans.utils_keras import KerasModelWrapper
#    wrap = KerasModelWrapper(model)
#
#
#    
## =============================================================================
##     Create ART classifier and test perturbations
## =============================================================================
#    classifier = SklearnClassifier(model=model)
#    
#    # Evaluate the ART classifier on benign test examples
#    predictions = classifier.predict(X_holdout)
#    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_holdout, axis=1)) / len(y_holdout)
#    print("Accuracy on benign test examples: {}%".format(accuracy * 100))
#    
#    # Evaluate on some train instances
#    x_pre_adv = X[0:10,:]
#    y_adv = y[0:10]
#    
#    predictions = classifier.predict(x_pre_adv)
#    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_adv, axis=1)) / len(y_adv)
#    print("Accuracy on instances before perturbation: {}%".format(accuracy * 100))
#    
#    # Generate adversarial test examples
#    attack = CarliniLInfMethod(estimator=classifier, confidence=0.9)
#    x_post_adv = np.rint(attack.generate(x=x_pre_adv))
#    
#    # Evaluate the ART classifier on adversarial test examples
#    predictions = classifier.predict(x_post_adv)
#    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_adv, axis=1)) / len(y_adv)
#    print("Accuracy on adversarial instances: {}%".format(accuracy * 100))











