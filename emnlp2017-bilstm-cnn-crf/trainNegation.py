# Under development 
# This script trains the BiLSTM-CRF architecture for token-level negation classification using
# a dev dataset for now. 
# The code use the embeddings by Komninos et al. (https://www.cs.york.ac.uk/nlp/extvec/)
from __future__ import print_function
import os
import logging
import sys
from neuralnets.BiLSTM import BiLSTM
from util.preprocessing import perpareDataset, loadDatasetPickle

from keras import backend as K

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
# datasets = {
#     'conll2000_chunking':                                   #Name of the dataset
#         {'columns': {0:'tokens', 1:'POS', 2:'chunk_BIO'},   #CoNLL format for the input data. Column 0 contains tokens, column 2 contains POS and column 2 contains chunk information using BIO encoding
#          'label': 'chunk_BIO',                                #Which column we like to predict
#          'evaluate': True,                                  #Should we evaluate on this task? Set true always for single task setups
#          'commentSymbol': None}                             #Lines in the input data starting with this string will be skipped. Can be used to skip comments
# }
datasets = {
    'i2b2_2010':                            #Name of the dataset
        {'columns': {0:'tokens', 3:'Section', 5:'Assertion'},   #CoNLL format for the input data. Column 1 contains tokens, column 3 contains POS information
         'label': 'Assertion',                     #Which column we like to predict
         'evaluate': True,                   #Should we evaluate on this task? Set true always for single task setups
         'commentSymbol': None}              #Lines in the input data starting with this string will be skipped. Can be used to skip comments
}

# :: Path on your computer to the word embeddings. Embeddings by Komninos et al. will be downloaded automatically ::
embeddingsPath = 'komninos_english_embeddings.gz'

# :: Prepares the dataset to be used with the LSTM-network. Creates and stores cPickle files in the pkl/ folder ::
pickleFile = perpareDataset(embeddingsPath, datasets)


######################################################
#
# The training of the network starts here
#
######################################################


#Load the embeddings and the dataset
embeddings, mappings, data = loadDatasetPickle(pickleFile)

# Some network hyperparameters
params = {'classifier': ['CRF'], 'LSTM-Size': [100, 100], 'dropout': (0.25, 0.25)}


model = BiLSTM(params)
model.setMappings(mappings, embeddings)
model.setDataset(datasets, data)
model.storeResults('results/i2b2_2010.csv') #Path to store performance scores for dev / test
model.modelSavePath = "models/[ModelName]_[DevScore]_[TestScore]_[Epoch].h5"
model.fit(epochs=25)



