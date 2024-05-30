#!/usr/bin/python
# This scripts loads a pretrained model and a input file in CoNLL format (each line a token, sentences separated by an empty line).
# The input sentences are passed to the model for tagging. Prints the tokens and the tags in a CoNLL format to stdout
# Usage: python RunModel_ConLL_Format.py modelPath inputPathToConllFile
# For pretrained models see docs/

from __future__ import print_function
import argparse
from util.preprocessing import readCoNLL, createMatrices, addCharInformation, addCasingInformation
from neuralnets.BiLSTM import BiLSTM
import os
import re
from util.casxmi2CONLL import FormatConvertor
from util.preprocessing import section_filter
import sys
import logging


# the target phrase type: choose from ExplicitGCSMention, ResponseRelatedToGCS, BarrierToGCSAssessment, AlternativeScale
GCS_PHRASE_TYPES = ['ExplicitGCSMention', 'ResponseRelatedToGCS', 'BarrierToGCSAssessment', 'AlternativeScale']

TYPE_SYSTEM = '/Users/chenkx/data/GCS/TypeSystem.xml'
BASE_MODEL_PATH = "models/base-adam-for-gcs_i2b2_2010_0.8693_0.8672_12.h5"
SECTION_MODEL_PATH = "models/with-section-adam-for-gcs_i2b2_2010_0.8712_0.8614_14.h5"

SECTION_COLUMNS = {0:'tokens', 3: 'section', 5:'concept'}
NO_SECTION_COLUMNS = {0:'tokens', 5:'concept'}


parser = argparse.ArgumentParser()
parser.add_argument(
    "--input-dir",
    dest="input_dir",
    type=str,
    default='',
    help="Input directory where CAS XMI annotations are stored",
)
# TODO: assert phrase type is supported 
parser.add_argument(
    "--phrase-type",
    dest="gcs_phrase_type",
    type=str,
    default='BarrierToGCSAssessment', 
    help="Input directory where CAS XMI annotations are stored",
)
args = parser.parse_args()


format_convertor = FormatConvertor( args.input_dir, 
                                   type_system_loc=TYPE_SYSTEM, 
                                   gcs_phrase_type=args.gcs_phrase_type)
input_conll = format_convertor.parse_text(verbose=False)

# sys.exit()

# :: Prepare the input ::
sentences = section_filter(readCoNLL("", NO_SECTION_COLUMNS, input_text=input_conll),  level=None)
addCharInformation(sentences)
addCasingInformation(sentences)


# :: Load the model ::
lstmModel = BiLSTM.loadModel(BASE_MODEL_PATH)


dataMatrix = createMatrices(sentences, lstmModel.mappings, True)

# :: Tag the input ::
predTags = lstmModel.tagSentences(dataMatrix)
correctTags = {}
for modelName in lstmModel.models:
    idx2Labels = lstmModel.idx2Labels[modelName]
    # correctTags[modelName] = [[label for label in sentences[idx][lstmModel.labelKeys[modelName]]] 
    #                         for idx in range(len(sentences))]


res = []
for sentenceIdx in range(len(sentences)):
    tokens = sentences[sentenceIdx]['tokens']

    for tokenIdx in range(len(tokens)):
        tags_to_print = []
        for modelName in sorted(predTags.keys()):
            # tags_to_print.append(correctTags[modelName][sentenceIdx][tokenIdx])
            tags_to_print.append(predTags[modelName][sentenceIdx][tokenIdx])

        res.append("\t".join(tags_to_print))
    res.append("")

splitted_input = input_conll.split('\n')
if len(splitted_input) > len(res):
    assert bool(re.fullmatch(r'\s+', splitted_input[-1]))
elif len(input_conll.split('\n')) < len(res): 
    assert bool(re.fullmatch(r'\s+', res[-1]))

outputs = []
for i in range(min(len(splitted_input), len(res))):
    outputs.append('\t'.join([splitted_input[i], res[i]]) )
with open("results/GCSpredictions/" + args.gcs_phrase_type + ".txt", "w") as f:
    f.writelines([line + "\n" for line in outputs])