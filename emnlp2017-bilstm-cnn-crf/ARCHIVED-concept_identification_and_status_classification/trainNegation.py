# Under development 
# This script trains the BiLSTM-CRF architecture for token-level negation classification using the i2b2 2010 dataset
# section labels are identified through regex 
# The code use the embeddings by Komninos et al. (https://www.cs.york.ac.uk/nlp/extvec/)

# arguments: 
#    section: default, no_section, both
    # default: train a negation model using section information 
    # no_section: train a negation model without adding section information 
    # both: do both
#    optimizer: adam, nadam, rmsprop, adadelta, adagrad, sgd


# Notes:
#   make sure to force create a new embedding 


from __future__ import print_function
import os
import logging
import sys
from neuralnets.BiLSTM import BiLSTM
from util.preprocessing import perpareDataset, loadDatasetPickle

from keras import backend as K


def train_negation(name, columns, force_create_new_embedding, optimizer, section_filter_level): 
    # :: Change into the working dir of the script ::
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    # :: Logging level ::
    loggingLevel = logging.INFO
    logger = logging.getLogger()
    logger.setLevel(loggingLevel)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(loggingLevel)
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


    ######################################################
    #
    # Data preprocessing
    #
    ######################################################
    datasets = {
        'i2b2_2010':                            #Name of the dataset
            {'columns': columns,   #CoNLL format for the input data. Column 1 contains tokens, column 3 contains POS information
            'label': 'Assertion_BIO',                     #Which column we like to predict
            'evaluate': True,                   #Should we evaluate on this task? Set true always for single task setups
            'commentSymbol': None}              #Lines in the input data starting with this string will be skipped. Can be used to skip comments
    }

    # :: Path on your computer to the word embeddings. Embeddings by Komninos et al. will be downloaded automatically ::
    embeddingsPath = 'komninos_english_embeddings.gz'

    # :: Prepares the dataset to be used with the LSTM-network. Creates and stores cPickle files in the pkl/ folder ::
    pickleFile = perpareDataset(embeddingsPath, datasets, forceNew=force_create_new_embedding, section_filter_level=section_filter_level)


    ######################################################
    #
    # The training of the network starts here
    #
    ######################################################


    #Load the embeddings and the dataset
    embeddings, mappings, data = loadDatasetPickle(pickleFile)

    # Some network hyperparameters
    params = {'classifier': ['CRF'], 'LSTM-Size': [100, 100], 'dropout': (0.25, 0.25), 'earlyStopping': 5, 'optimizer': optimizer}


    model = BiLSTM(params)
    model.setMappings(mappings, embeddings)
    model.setDataset(datasets, data)
    model.storeResults('results/' + name + '_i2b2_2010.csv') #Path to store performance scores for dev / test
    model.modelSavePath = "models/"+ name + "_[ModelName]_[DevScore]_[TestScore]_[Epoch].h5"
    model.fit(epochs=25)


######################################################
#
# Set parameters
#
######################################################
cols_no_section = {0:'tokens', 5:'Assertion_BIO'}
cols_with_section = {0:'tokens', 3: 'section', 5:'Assertion_BIO'}

print("\n\n============================\n No sections - nadam\n============================\n")
train_negation(name = 'no-section-nadam', 
               columns = cols_no_section, 
               force_create_new_embedding = True, 
               optimizer = 'nadam',
               section_filter_level = None)

print("\n\n============================\n No sections - adadelta\n============================\n")
train_negation(name = 'no-section-adadelta', 
               columns = cols_no_section, 
               force_create_new_embedding = False, 
               optimizer = 'adadelta',
               section_filter_level = None)

print("\n\n============================\n No sections - adagrad\n============================\n")
train_negation(name = 'no-section-adagrad', 
               columns = cols_no_section, 
               force_create_new_embedding = False, 
               optimizer = 'adagrad',
               section_filter_level = None)

print("\n\n============================\n No sections - nadam\n============================\n")
train_negation(name = 'one-model-with-section-nadam', 
               columns = cols_with_section, 
               force_create_new_embedding = True, 
               optimizer = 'nadam',
               section_filter_level = None)

print("\n\n============================\n No sections - adadelta\n============================\n")
train_negation(name = 'one-model-with-section-adadelta', 
               columns = cols_with_section, 
               force_create_new_embedding = False, 
               optimizer = 'adadelta',
               section_filter_level = None)

print("\n\n============================\n No sections - adagrad\n============================\n")
train_negation(name = 'one-model-with-section-adagrad', 
               columns = cols_with_section, 
               force_create_new_embedding = False, 
               optimizer = 'adagrad',
               section_filter_level = None)





