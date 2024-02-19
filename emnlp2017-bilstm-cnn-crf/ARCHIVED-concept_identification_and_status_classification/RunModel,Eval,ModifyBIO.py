#!/usr/bin/python
# This scripts loads a pretrained model and a input file in CoNLL format (each line a token, sentences separated by an empty line).
# The input sentences are passed to the model for tagging. Prints the tokens and the tags in a CoNLL format to stdout
# Usage: python RunModel_ConLL_Format.py modelPath inputPathToConllFile
# For pretrained models see docs/
from __future__ import print_function
from util.preprocessing import readCoNLL, createMatrices, addCharInformation, addCasingInformation
from neuralnets.BiLSTM import BiLSTM
from util.preprocessing import section_filter
import sys
import logging


if len(sys.argv) < 3:
    print("Usage: python RunModel,Eval,ModifyBIO.py modelPath inputPathToConllFile outputCSVPath")
#     exit()

# modelPath = sys.argv[1]
# inputPath = sys.argv[2]
# outputPath = sys.argv[3]
modelPath = "models/no_section_adam_i2b2_2010_0.7909_0.7794_16.h5"
inputPath = "data/i2b2_2010/test.txt"
inputColumns = {0:'tokens', 3: 'section', 5:'Assertion_BIO'}

filter_section_level = "high" # "high", "moderate", "low", None
if filter_section_level in ["high", "moderate", "low"]: 
    print("Filter to contain %sly-negated sections" % filter_section_level)

# :: Prepare the input ::
sentences = section_filter(readCoNLL(inputPath, inputColumns),  filter_section_level)
addCharInformation(sentences)
addCasingInformation(sentences)


# :: Load the model ::
lstmModel = BiLSTM.loadModel(modelPath)


dataMatrix = createMatrices(sentences, lstmModel.mappings, True)

# :: Tag the input ::
tags = lstmModel.tagSentences(dataMatrix)


# :: Output to stdout ::
for sentenceIdx in range(len(sentences)):
    tokens = sentences[sentenceIdx]['tokens']

    for tokenIdx in range(len(tokens)):
        tokenTags = []
        for modelName in sorted(tags.keys()):
            tokenTags.append(tags[modelName][sentenceIdx][tokenIdx])

        print("%s\t%s" % (tokens[tokenIdx], "\t".join(tokenTags)))
    print("")