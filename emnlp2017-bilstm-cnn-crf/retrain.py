# This script trains the BiLSTM-CRF architecture for part-of-speech tagging
# and stores it to disk. Then, it loads the model to continue the training.
# For more details, see docs/Save_Load_Models.md
from __future__ import print_function
import os
import logging
import sys
from neuralnets.BiLSTM import BiLSTM
from util.preprocessing import perpareDataset, loadDatasetPickle



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
        "i2b2_2010_lowly_negated":                            #Name of the dataset
            {'columns': {0:'tokens', 5:'concept', 6:'Assertion'},   #CoNLL format for the input data. Column 1 contains tokens, column 3 contains POS information
            'label': 'Assertion',                     #Which column we like to predict
            'evaluate': True,                   #Should we evaluate on this task? Set true always for single task setups
            'commentSymbol': None}              #Lines in the input data starting with this string will be skipped. Can be used to skip comments
    }


# :: Path on your computer to the word embeddings. Embeddings by Komninos et al. will be downloaded automatically ::
embeddingsPath = 'komninos_english_embeddings.gz'

# :: Prepares the dataset to be used with the LSTM-network. Creates and stores cPickle files in the pkl/ folder ::
pickleFile = perpareDataset(embeddingsPath, datasets, forceNew=False, section_filter_level=None)


######################################################
#
# The training of the network starts here
#
######################################################


#Load the embeddings and the dataset
embeddings, mappings, data = loadDatasetPickle(pickleFile)

# Some network hyperparameters
params = {'classifier': ['CRF'], 'LSTM-Size': [100, 100], 'dropout': (0.25, 0.25), 'earlyStopping': 10, 'optimizer': 'adam'}


print("Load the model and continue training")
newModel = BiLSTM.loadModel("/scratch/kexin/clinical_negation/LSTMmodels/no-section-adam_i2b2_2010_lowly_negated_0.8862_0.8762_11.h5")
newModel.setDataset(datasets, data)
newModel.storeResults('results/no-section-adam_i2b2_2010_lowly_negated2.csv')
newModel.modelSavePath = "/scratch/kexin/clinical_negation/LSTMmodels/no-section-adam_[ModelName]_[DevScore]_[TestScore]_[Epoch].h5"
newModel.fit(epochs=14)


