# This script is based on the notebooks/2010Corpus/explore_i2b2-2010-v3.0_train.ipynb
# This scrip is function based. Not object oriented design.

##### debug log #####
# To identify which section a concept falls in, I mistakenly used `bisect_left` in previous scripts rather than `bisect`


import os
import pandas as pd
import numpy as np
import re
import json
import bisect
import nltk
import csv

######## draft codes ###############
# print(bisect.bisect([0, 10, 20], 5))
# print(bisect.bisect([0, 10, 20], 10))
# print(bisect.bisect([0, 10, 20], 20))
# print(bisect.bisect([0, 10, 20], 21))
# print(bisect.bisect([0, 10, 20], -1))
# ############################### 



DATA_DIR_RAW = '/Users/chenkx/git/clinical-negation/Negation/data/2010_relations_challenge'
DATA_DIR = r'/Users/chenkx/Box Sync/NLP group/2010 i2b2 challenge - rel'
FILE_SORTER= r"/Users/chenkx/Box Sync/NLP group/2019 n2c2 Challenge/Track 3 (normalization)/Test/test_file_list.txt"
MAP_DIR = r"/Users/chenkx/git/clinical-negation/notebooks/2010Corpus/section_mapping_v4_all.csv"
MAP_DIR1 = "/Users/chenkx/git/clinical-negation/data/simple_header_map.json"
MAP_DIR2 = "/Users/chenkx/Desktop/TBIC-not_synced/Sectionizer/data/section_mapping/sectionTypeMapping.json"



def header_pattern(txt):
    """
    Return an iterator yielding match objects over all non-overlapping matches
    """
    return re.finditer('(?<=\n)[a-zA-Z -]+(?=[ ]:[\n| ])', txt) # wrong and previously used: (\n[a-zA-Z -]+)(( :\n)|( : ))

def std_header(phrase):
    """
    standardize heading from regex matches.
        1. converting to lower case
        2. trim white space.
    If mapped them to "?": do not consider them as headers 
    If mapped to subsection, date/time, or providers, do not consider them as headers 
    """        

    # Load section heading map and prepare functions to extract headings
    with open(MAP_DIR, 'r') as f:
        section_map = f.read()
    section_map = section_map.split('\n')[1:]
    section_map = {i.split(',')[0]:i.split(',')[1] for i in section_map}

    with open(MAP_DIR1, 'r') as f:
        section_map1 = json.load(f)
    with open(MAP_DIR2, 'r') as f:
        section_map2_tmp = json.load(f)
    section_map2 = {}
    # reformat section_map2 to lower case
    for i in section_map2_tmp:
        section_map2[i.lower()] = section_map2_tmp[i].lower()
    # del section_map2_tmp

    phrase0 = re.sub(" :$", "", phrase.strip().lower())
    phrase = None
    if phrase0 in section_map1:
        phrase = section_map1[phrase0]
    elif phrase0 in section_map2:
        phrase = section_map2[phrase0]
    if not phrase:
        phrase = phrase0
    if phrase in section_map:
        phrase = section_map[phrase]
    else:
        print(f'Not mapped: {phrase0}')
        return None
        
    if phrase == '?':
        return None

    if phrase == '':
        return None
    
    if phrase == "Subsection" or phrase == "Date/Time" or phrase == "Providers":
        return None
    
    return phrase

def get_all_headings(txt):
    """
    1. Extract regex headings. 
    2. Map them to the normalized section according to the section map
    3. Sort headings based on begin offsets
    :return List[(str, int, int)] 
    """
    all_headings = []
    matches = header_pattern(txt)
    for m in matches:
        match = std_header(m.group(0))
        if match:
            b, e = m.span()
            all_headings.append( (match, b, e) )
    all_headings.sort(key=lambda x:x[1])
    return all_headings

def get_section(x, section_delim, headings):
    s_i = bisect.bisect_left(section_delim, x) - 1 
    if s_i == -1:
        section = 'Unknown/Unclassified'
    else:
        section = [i[0] for i in headings][s_i]
    if section == "?":
        section = "Unknown/Unclassified"
    return section

def get_concept(b, e, txt):
    return txt[b:e+1]

class Reader:
    def __init__(self, path, fname):
        """
        :param path - Path to the folder of which subfolders include "txt" and "ref"
        """

        self._path = path
        self.fname = fname
        
        self.ann = {}
        self._ann_raw = []
        self.all_headings = []
        
        with open(os.path.join(self._path, "txt", self.fname+".txt"), 'r') as f:
            self.txt = f.read()
        
    def get_all_headings(self):
        matches = header_pattern(self.txt)
        for m in matches:
            match = std_header(m.group(0))
            if match:
                b, e = m.span()
                self.all_headings.append( (match, b, e) )
#         self.all_headings = [std_header(match) for i, match in enumerate([re.search('[a-zA-Z ]+(( :$)|( : ))', txt) for txt in self.txt.split('\n')])]
    
    def get_annotation(self, must_have_assertion=True):
        """
        {
            fname: {
                iterm_id: {
                    b: int begin_offset, 
                    e: int end_offset, 
                    t: str "type",
                    a: str "assertion", 
                    c: str "concept_raw_text", 
                    s: None, 
                    sent: None
                }
                length: int length of the note 
            }
        }

        """
        fname = self.fname
        with open(os.path.join(self._path, "ref", fname+".ann"), 'r') as f:
            ann_raw = f.read().split('\n')
            self._ann_raw = ann_raw
        
        annotations = {}
        for line in ann_raw:
            line = line.split('\t')
            if line[0].startswith('T'):
                annotations[line[0]] = {
                    'b': int(line[1].split()[1]),  
                    'e': int(line[1].split()[2]), 
                    't': line[1].split()[0], 
                    'a': None, 
                    'c': line[2], 
                    's': None, 
                    'sent': None
                }
        for line in ann_raw:
            line = line.split('\t')
            if line[0].startswith('A'):
                if line[1].split()[1] not in annotations:
                    print(f"Warning: {line[0]} ??")
                annotations[line[1].split()[1]]['a'] = line[1].split()[0]
                
        # remove annotations that don't have assertion informaiton 
        if must_have_assertion:
            delete = []
            for i in annotations:
                if annotations[i]['a'] is None:
                    delete.append(i)
            for i in delete:
                del annotations[i]
        
        annotations['length'] = len(self.txt)
        
        self.ann = {fname: annotations}    

filenames_train = [i[:-4] for i in os.listdir(os.path.join(\
    DATA_DIR, "train", "txt")) if i.endswith(".txt")]
filenames_test = [i[:-4] for i in os.listdir(os.path.join(\
    DATA_DIR, "test", "txt")) if i.endswith(".txt")]

# temporary: includes only partial files
with open(FILE_SORTER, 'r') as f:
    sorter = f.read().split('\n')
sorter = [i for i in sorter if i != ""]
print('This script only uses partial of the 2010 challenge data')
filenames_test = [i for i in filenames_test if i in sorter]
filenames_train = [i for i in filenames_train if i in sorter]
###########

####### temporary codes ############
# reader = Reader(os.path.join(DATA_DIR, 'train'), '134300717')
# tokenizer = nltk.tokenize.PunktSentenceTokenizer()
# sent_text = tokenizer.span_tokenize(reader.txt)
# for i in sent_text:
#     print(f'====== {i[0]} : {i[1]} ========\n{reader.txt[i[0]:i[1]]}')
#############################


all_annot = {}
which_set = "train" # changed to test at some point
# Get annotations from the reference. Section info is not included yet.
for i, file in enumerate(filenames_train + filenames_test):
    if i == len(filenames_train):
        which_set = "test"
    reader = Reader(os.path.join(DATA_DIR, which_set), file)
    reader.get_annotation()
    if not reader.ann:
        print(f'{file} does not contain assertion annotations')
        continue
    all_annot = {**all_annot, **reader.ann}

for f in all_annot:
    try:
        reader = Reader(os.path.join(DATA_DIR, "train"), f)
    except FileNotFoundError:
        reader = Reader(os.path.join(DATA_DIR, "test"), f)
    reader.get_all_headings()
    h_left = [i[1] for i in reader.all_headings]

    #### New: split sentences ####
    sent_splitter = nltk.tokenize.PunktSentenceTokenizer().span_tokenize(reader.txt)
    sent_bound = [(s, e) for s, e in sent_splitter]
    sent_bound.sort(key=lambda x:x[0])
    sent_left = [i[0] for i in sent_bound]
    ################

    for i in all_annot[f]:
        if i == 'length':
            continue
        annotation = all_annot[f][i]

        s_i = bisect.bisect_right(h_left, all_annot[f][i]['b']) - 1 # TODO: this line can be optimized
        if s_i == -1:
            section = 'Unknown/Unclassified'
        else:
            section = [i[0] for i in reader.all_headings][s_i]
        all_annot[f][i]['s'] = section

        #### New: get sentence ####
        sent_i = bisect.bisect_right(sent_left, all_annot[f][i]['b']) - 1 # TODO: this line can be optimized
        if sent_i == -1:
            print(f'{f}')
            break
        else:
            b, e = sent_bound[sent_i]
            all_annot[f][i]['sent'] = reader.txt[b:e]
        #####################

df = pd.DataFrame(columns=['b', 'e', 't', 'a', 'c', 's', 'sent'])
for f in all_annot:
    for i in all_annot[f]:
        if i == 'length':
            continue
        tmp =pd.DataFrame(all_annot[f][i], index=['::'.join([f, i])])
        df = pd.concat([df, tmp], ignore_index=False)

# df.to_csv("data/extracted_concepts/explore_i2b2-2010-problem_concepts-with_sentences_partial.csv", index_label="id")