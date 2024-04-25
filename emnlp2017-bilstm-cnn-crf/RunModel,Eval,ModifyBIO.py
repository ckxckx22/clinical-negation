#!/usr/bin/python
# This scripts loads a pretrained model and a input file in CoNLL format (each line a token, sentences separated by an empty line).
# The input sentences are passed to the model for tagging. Prints the tokens and the tags in a CoNLL format to stdout
# Usage: python RunModel_ConLL_Format.py modelPath inputPathToConllFile
# For pretrained models see docs/
from __future__ import print_function
from util.preprocessing import readCoNLL, createMatrices, addCharInformation, addCasingInformation
from neuralnets.BiLSTM import BiLSTM
import os
from util.preprocessing import section_filter
import sys
import logging


# if len(sys.argv) < 3:
#     print("Usage: python RunModel,Eval,ModifyBIO.py modelPath inputPathToConllFile outputCSVPath")
#     exit()

# modelPath = sys.argv[1]
# inputPath = sys.argv[2]
# outputPath = sys.argv[3]
    
best_models = {
    # "base-adam":"base-adam_i2b2_2010_0.8719_0.8677_17.h5",
    # "section-adam": "section-adam_i2b2_2010_0.8810_0.8681_18.h5",
    # "base-nadam":"no-section-nadam_i2b2_2010_0.8754_0.8684_13.h5",
    # "section-nadam": "section-nadam_i2b2_2010_0.8705_0.8705_13.h5",
    # "highly-negated-base-adam": "adam_i2b2_2010_highly_negated_0.8299_0.7958_22.h5",
    # "highly-negated-with-section-adam": "specific-sections-adam_i2b2_2010_highly_negated_0.8504_0.8119_14.h5",
    # "lowly-negated-base-adam": "no-section-adam_i2b2_2010_lowly_negated_0.8862_0.8762_11.h5",
    # "lowly-negated-with-section-adam":"specificly-sections-adam_i2b2_2010_lowly_negated_0.8664_0.8577_26.h5",
    # "downsample-lowly-negated-base-adam":"no-section-adam_i2b2_2010_downsample-lowly_negated_0.7309_0.7105_8.h5",
    # "downsample-lowly-negated-with-section-adam":"specificly-sections-adam_i2b2_2010_downsample-lowly_negated_0.7500_0.7266_12.h5",
    # "downsample-base":"no-section-adam_i2b2_2010_downsample_0.6208_0.5737_3.h5",
    # "downsample-with-section":"specificly-sections-adam_i2b2_2010_downsample_0.7754_0.7596_15.h5",
    "show-highly-negated-base": "shwoing-highly-negated-base_i2b2_2010-showing-highly_negated_0.7405_0.6913_12.h5",
    "show-highly-negated-with_section": "shwoing-highly-negated-with-sections_i2b2_2010-showing-highly_negated_0.7698_0.7276_9.h5",
    "show-lowly-negated-base": "shwoing-lowly-negated-base_i2b2_2010-showing-lowly_negated_0.8448_0.8464_13.h5",
    "show-lowly-negated-with_section": "shwoing-lowly-negated-with-sections_i2b2_2010-showing-lowly_negated_0.8536_0.8442_19.h5"
}

input_paths = {
    # "highly-negated-base-adam": "data/i2b2_2010_highly_negated/test.txt",
    # "highly-negated-with-section-adam": "data/i2b2_2010_highly_negated/test.txt",
    # "lowly-negated-base-adam": "data/i2b2_2010_lowly_negated/test.txt",
    # "lowly-negated-with-section-adam":"data/i2b2_2010_lowly_negated/test.txt",
    # "downsample-lowly-negated-base-adam":"data/i2b2_2010_downsample-lowly_negated/test.txt",
    # "downsample-lowly-negated-with-section-adam":"data/i2b2_2010_downsample-lowly_negated/test.txt",
    # "downsample-base": "data/i2b2_2010_downsample/test.txt",
    # "downsample-with-section":"data/i2b2_2010_downsample/test.txt",
    "show-highly-negated-base": "data/i2b2_2010-showing-highly_negated/test.txt",
    "show-highly-negated-with_section": "data/i2b2_2010-showing-highly_negated/test.txt",
    "show-lowly-negated-base": "data/i2b2_2010-showing-lowly_negated/test.txt",
    "show-lowly-negated-with_section": "data/i2b2_2010-showing-lowly_negated/test.txt"
}

section_cols = {0:'tokens', 3: 'section', 5:'concept', 6:'Assertion'}
no_section_cols = {0:'tokens', 5:'concept', 6:'Assertion'}

columns = {
    # "highly-negated-base-adam": no_section_cols,
    # "highly-negated-with-section-adam": section_cols,
    # "lowly-negated-base-adam": no_section_cols,
    # "lowly-negated-with-section-adam": section_cols,
    # "downsample-lowly-negated-base-adam": no_section_cols,
    # "downsample-lowly-negated-with-section-adam": section_cols,
    # "downsample-base": no_section_cols,
    # "downsample-with-section": section_cols,
    "show-highly-negated-base": no_section_cols,
    "show-highly-negated-with_section": section_cols,
    "show-lowly-negated-base": no_section_cols,
    "show-lowly-negated-with_section": section_cols,

}


# MODELNAME = "downsample-lowly-negated-with-section-adam"
for MODELNAME in best_models:
    if MODELNAME == "highly-negated-base-adam":
        continue
    print(MODELNAME)

    modelPath = os.path.join("/scratch/kexin/clinical_negation/LSTMmodels/save",best_models[MODELNAME])
    inputPath = input_paths[MODELNAME]
    inputColumns = columns[MODELNAME]

    filter_section_level = None # "high", "moderate", "low", None
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
    predTags = lstmModel.tagSentences(dataMatrix)
    correctTags = {}
    for modelName in lstmModel.models:
        idx2Labels = lstmModel.idx2Labels[modelName]
        correctTags[modelName] = [[label for label in sentences[idx][lstmModel.labelKeys[modelName]]] 
                                for idx in range(len(sentences))]

    # :: Output to stdout ::
    res = []
    for sentenceIdx in range(len(sentences)):
        tokens = sentences[sentenceIdx]['tokens']

        for tokenIdx in range(len(tokens)):
            tags_to_print = []
            for modelName in sorted(predTags.keys()):
                # tags_to_print.append(correctTags[modelName][sentenceIdx][tokenIdx])
                tags_to_print.append(predTags[modelName][sentenceIdx][tokenIdx])
                tags_to_print.append(correctTags[modelName][sentenceIdx][tokenIdx])

            res.append("%s\t%s" % (tokens[tokenIdx], "\t".join(tags_to_print)))
        res.append("")
    with open("results/predictions-" + MODELNAME + ".txt", "w") as f:
        f.writelines([line + "\n" for line in res])