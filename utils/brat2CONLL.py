# This script transforms brat annotations to CONLL format.
# The target BIO-format labels represents assertion status
# the resulting output will not remove punctuations or white spaces.
# This script takes two or three arguments from command line: data_directory, output_directory, (filename)

# Output format: Token, begin_offset, end_offset, section_type, file_name, label (i.e. assertion_status)

# Example output
# ulcerative	520	530	Present illness	0001.txt	B-present
# colitis	531	538	Present illness	0001.txt	I-present
# on	539	541	Present illness	0001.txt	O
# Asacol	542	548	Present illness	0001.txt	O
# presents	549	557	Present illness	0001.txt	O
# with	558	562	Present illness	0001.txt	O
# brbpr	563	568	Present illness	0001.txt	B-present
# starting	569	577	Present illness	0001.txt	O
# at	578	580	Present illness	0001.txt	O
# 9am	581	584	Present illness	0001.txt	O
# of	585	587	Present illness	0001.txt	O
# the	588	591	Present illness	0001.txt	O
# morning	592	599	Present illness	0001.txt	O

# Note that this script uses py3.8. Using py3.9 may cause issues with TreebankWordTokenizer().span_tokenize(text)
# To use this scripts for other purposes than "Assertion status", check and modify accordingly all codes commented as
# "#FILTER"


import json
import re
from bisect import bisect_left
from collections import deque
import threading
import concurrent.futures
import os
import sys
from typing import Optional, List, Tuple, Dict, Deque, Iterator, Match
import warnings

from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize import TreebankWordTokenizer


ALL_UNIQUE_LABELS = ["present", "absent", "possible", "conditional", "hypothetical", "associated_with_someone_else"]
DEFAULT_DATA_DIR = r'/Users/chenkx/Box Sync/NLP group/2010 i2b2 challenge - rel/train'
DEFAULT_OUTPUT_DIR = os.path.join(DEFAULT_DATA_DIR, "CONLL")
DEFAULT_MAP_DIR = "/Users/chenkx/git/clinical-negation/notebooks/2010Corpus/section_mapping_v4_all.csv"
ADDITIONAL_MAP_DIR1 = "/Users/chenkx/git/clinical-negation/data/simple_header_map.json"
ADDITIONAL_MAP_DIR2 = "/Users/chenkx/Desktop/TBIC-not_synced/Sectionizer/data/section_mapping/sectionTypeMapping.json"
HEADER_PATTERN = "(?<=\n)[a-zA-Z -]+(?=[ ]:[\n| ])"


def get_dir():
    """
    return input_data_dir, output_data_dir, [file_name]
    """
    input_data_dir: str
    output_data_dir: str
    fname: Optional[str] = None
    if len(sys.argv) == 1:
        input_data_dir = DEFAULT_DATA_DIR
        output_data_dir = DEFAULT_OUTPUT_DIR
    elif len(sys.argv) == 3 or len(sys.argv) == 4:
        input_data_dir = sys.argv[1]
        output_data_dir = sys.argv[2]
    elif len(sys.argv) == 4:
        input_data_dir = sys.argv[1]
        output_data_dir = sys.argv[2]
        fname = sys.argv[3]
    else:
        sys.exit("Need either zero, two, or three arguments")
    if not os.path.exists(input_data_dir):
        sys.exit("input_dir not valid: " + input_data_dir)
    if not os.path.exists(output_data_dir):
        print("Creating output directory...")
        os.makedirs(output_data_dir, exist_ok=True)
    return input_data_dir, output_data_dir, fname


class Token:
    """
    TreebankWordTokenizer is used to split span into tokens
    Note: white space will not be trimmed
    """

    def __init__(self, begin: int, end: int, label: str,
                 token: Optional[str] = None, section: str = "N/A", raw_note: Optional[str] = None):
        self.token: Optional[str] = None
        if token:
            self.token = token
        elif raw_note:
            self.token = raw_note[begin: end]
        self.b: int = begin
        self.e: int = end
        self.label: str = label
        Token.validate_IBOlabel(label)
        self.sec: str = section

    def change_section_to(self, section: str):
        self.sec = section

    def change_label_to(self, new: str):
        self.label = new
        Token.validate_IBOlabel(new)

    @staticmethod
    def validate_IBOlabel(label):
        """
        label must be IBO-encoded and be one of ALL_UNIQUE_LABELS
        """
        assert label.startswith("I-") or label.startswith("B-") or label == "O"
        if label != "O" and label[2:] not in ALL_UNIQUE_LABELS:
            warnings.warn(f"{file_name}: Unknown label type: {label}")

    @staticmethod
    def sentence_end():
        """
        A placeholder to represent a "sentence end". When printing to CONLL format, will create an empty line.
        :return: an empty Token
        """
        return Token(-1, -1, "O")

    def is_sentence_end(self) -> bool:
        if self.b == -1 and self.e == -1:
            return True
        return False

    def to_conll_line(self, fname: str = "_") -> str:
        if self.is_sentence_end():
            return ""
        else:
            if "" in [self.token, str(self.b), str(self.e), self.sec, fname, self.label]:
                warnings.warn("wrong")
                print([self.token, str(self.b), str(self.e), self.sec, fname, self.label])
            return "\t".join([self.token, str(self.b), str(self.e), self.sec, fname, self.label])


class Span:
    """
    A span of text. A span is NOT a sentence!
    """
    def __init__(self, begin: int, end: int, label: str, section: str = "N/A"):
        self.b: int = begin
        self.e: int = end  # exclusive indexing
        self.label: Optional[str] = label
        self.sec: str = section

    def tokenize(self, note: str) -> List[Token]:
        """
        Tokenize a span.
        :return tokens sorted by beginning offsets.
        """
        text: str = note[self.b: self.e]
        bio_label: str = "O"
        # BIO format: add an "I-" prefix for all tokens. (later will change it for the first token to "B-")
        if self.label != "O":
            bio_label = "I-" + self.label
        tokens: List[Token] = [Token(b + self.b, e + self.b, bio_label, raw_note=note)
                               for b, e in TreebankWordTokenizer().span_tokenize(text)]
        tokens.sort(key=lambda x: x.b)
        # BIO format: for the first token, add "B-" prefix
        if self.label != "O":
            tokens[0].change_label_to("B-" + self.label)
        return tokens


def read_and_parse(file: str, data_dir: str) -> Tuple[str, List[Span]]:
    """
    Parse brat annotations (.ann) and the corresponding raw texts
    :param file
    :param data_dir
    :return: raw_text, Concepts
    """

    def read_annotation_note(file: str, data_dir: str) -> Tuple[str, str]:
        """
        Read .ann and .txt files of the given filename
        :param file:
        :param data_dir: directory to both files
        :return:
        """
        with open(os.path.join(data_dir, "txt", file + ".txt"), 'r') as f_to_read:
            content = f_to_read.read()
        with open(os.path.join(data_dir, "ref", file + ".ann"), 'r') as f_to_read:
            annotation = f_to_read.read()
        return content, annotation

    def parse_annotation(raw_annotation: str) -> List[Span]:
        """
        Parse .ann format string to get key info such as offsets and labels (e.g. assertion type) of concepts
        :param raw_annotation: raw annotations (.ann format)
        :return:
        """
        raw_annotation_lines: List[str] = raw_annotation.split('\n')
        concepts_with_id: Dict[str:Span] = {}

        # find all annotations of "problem"
        for raw_line in raw_annotation_lines:
            line: List[str] = raw_line.split('\t')
            if line[0].startswith('T'):
                # only concepts of "problem" have assertion status
                if line[1].split()[0] != "problem":  #FILTER
                    continue
                concepts_with_id[line[0]] = Span(int(line[1].split()[1]),  # start offset
                                                 int(line[1].split()[2]),  # end offset
                                                 "placeholder")  # label
        # find all annotations of labels (i.e. assertion_status)
        for raw_line in raw_annotation_lines:
            line: List[str] = raw_line.split('\t')
            if line[0].startswith('A'):
                concept_id: str = line[1].split()[1]
                if concept_id not in concepts_with_id:
                    print(f"Warning: {line[0]} ??")
                concepts_with_id[concept_id].label = line[1].split()[0]

        # Check whether the label "placeholder" of every concept has been updated.
        concepts: List[Span] = list(concepts_with_id.values())
        for c in concepts:
            if c.label == "placeholder":
                warnings.warn("The label for a concept is not specified. begin= {c.b}, end = {c.e}")
                c.label = "O"  # Mark this annotation as a non-concept

        return concepts

    text, annotations = read_annotation_note(file, data_dir)
    return text, parse_annotation(annotations)


def spans_not_overlapping(spans: List[Span]) -> bool:
    """
    Make sure spans are not overlapping
    :param spans: spans sorted according to their beginning offsets
    """
    prev_ending = -1
    for sp in spans:
        if sp.b < prev_ending:
            return False
        else:
            prev_ending = sp.e
    return True


def regex_section_finder(raw_note: str) -> Dict[int, str]:
    """
    Find section headers from a note using regex and a map to normalize the found headings
    """
    # get all headings
    all_headings: Dict[int, str] = {}  # {begin_offset: heading}
    # an iterator yielding match objects over all non-overlapping matches
    matches: Iterator[Match[str]] = re.finditer(HEADER_PATTERN, raw_note)  # TODO: ADD PARENTHESIS HERE
    for m in matches:
        match: Optional[str] = SectionFinder.std_header(m.group(0))
        if match:
            b, e = m.span()
            all_headings[b] = match
    if len(all_headings) == 0:
        warnings.warn("%s: No sections identified using regex." % file_name)
    return all_headings


class SectionFinder:
    """
    Identify a span's section type. One SectionFinder is created for each note.
    """
    def __init__(self, raw_note: str):
        # all headings and their beginning offsets
        self.all_headings: Dict[int: str]
        # find sections using regex
        self.all_headings = regex_section_finder(raw_note)

    def get_section_type(self, span: Optional[Span] = None, token: Optional[Token] = None) -> str:
        """
        provide either a span or a token to get its section type
        """
        if span is None and token is None:
            warnings.warn("%s: Provide value to get section type!" % file_name)
            return ""

        if len(self.all_headings) == 0:
            return 'Unknown/Unclassified'

        # the beginning offsets of all sections
        section_beginnings: List[int] = list(self.all_headings.keys())
        section_beginnings.sort()

        begin: int
        if span:
            begin = span.b
        else:
            begin = token.b
        idx: int = bisect_left(section_beginnings, begin) - 1
        section: str
        if idx == -1:
            section = 'Unknown/Unclassified'
        else:
            section = self.all_headings[section_beginnings[idx]]
        if section is None or section == "":
            warnings.warn("Section not found!!")
        return section

    @staticmethod
    def std_header(phrase: str, section_map_dir: str = DEFAULT_MAP_DIR) -> Optional[str]:
        """
        standardize heading from regex matches.
            1. converting to lower case
            2. trim white space.
        If a token is mapped to "?": consider it as non-headers.
        "Subsection", "Date/Time", "Providers" are NOT valid section types.
        """
        # Load map
        if not os.path.exists(section_map_dir):
            warnings.warn(f"Section map ({section_map_dir}) does not exist. "
                          f"Load section map from default directory: " + DEFAULT_MAP_DIR)
        with open(section_map_dir, 'r') as f:
            raw_section_map: List[str] = f.read().split('\n')[1:]
        section_map: Dict[str: str] = {i.split(',')[0]: i.split(',')[1] for i in raw_section_map}

        # Load additional maps
        with open(ADDITIONAL_MAP_DIR1, 'r') as f:
            section_map1: Dict[str: str] = json.load(f)
        with open(ADDITIONAL_MAP_DIR2, 'r') as f:
            section_map2_tmp: Dict[str: str] = json.load(f)
        section_map2: Dict[str: str] = {}
        # reformat section_map2 to lower case
        for i in section_map2_tmp:
            section_map2[i.lower()] = section_map2_tmp[i].lower()
        del section_map2_tmp

        phrase0: str = re.sub(" :$", "", phrase.strip().lower())
        result: str = phrase0
        if phrase0 in section_map1:
            result = section_map1[phrase0]
        elif phrase0 in section_map2:
            result = section_map2[phrase0]
        if result in section_map:
            result = section_map[result]
        else:
            print(f'Not mapped: {phrase0}')
            return None

        if result in ['?', '', "Subsection", "Date/Time", "Providers"]:
            return None

        return result


def tokenize_and_classify_section(span: Span, note: str, section_finder: SectionFinder) -> List[Token]:
    tokens: List[Token] = span.tokenize(note)
    for tk in tokens:
        if tk.is_sentence_end():
            continue
        tk.change_section_to(section_finder.get_section_type(token=tk))
    return tokens


def run(filename: str) -> None:
    """
    How it works:
    For each note,
    1. Parse annotations and the raw text to get the offsets and label for the concepts
    2. Identify sections using regex
    3. Slice the raw text into spans (i.e. span - concept - span - ...). Assign a label of "O" to all non-concepts
    4. Identify sentences: Add empty tokens to indicate end of sentences.
    5. Convert all tokens to CoNLL lines.
    """
    if filename not in wanted:
        return

    print(filename)

    raw_text, concepts = read_and_parse(filename, input_dir)
    # move on to next note if no annotations
    if not concepts:
        pass

    # identify sections using regex
    section_finder: SectionFinder = SectionFinder(raw_text)

    # Make sure concepts don't overlap
    concepts.sort(key=lambda x: x.b)
    assert spans_not_overlapping(concepts)

    # get all spans (a span is either a concept or the span of text between two concepts)
    all_spans: List[Span] = []
    previous_span_start: int = 0
    for c in concepts:
        if previous_span_start < c.b:
            all_spans.extend([Span(previous_span_start, c.b, "O"), c])
            previous_span_start = c.e
        elif previous_span_start == c.b:
            all_spans.append(c)
        else:
            warnings.warn("%s: Wrong... previous_span_start?" % file_name)
    # append the last span, if any
    if previous_span_start < len(raw_text):
        all_spans.append(Span(previous_span_start, len(raw_text), "O"))

    # Annotate each token in each span. Use a Deque instead of List because the need of popleft() later
    all_tokens: Deque['Token'] = deque([])
    for sp in all_spans:
        # all_tokens.extend(sp.tokenize(raw_text)) # if not want section information
        all_tokens.extend(tokenize_and_classify_section(sp, raw_text, section_finder))

    # get and sort all sentences (the offsets of spans)
    sentences: List[List[int]] = list(PunktSentenceTokenizer().span_tokenize(raw_text))
    # find all tokens within each sentence
    all_tokens_with_sentence_endings = []
    # Assume the ending offset of the current sentence = the beginning offset of the following sentence - 1,
    # skip the first sentence. Will start from the second sentence
    for sent in sentences[1:]:
        ending_index: int = sent[1] - 1
        while all_tokens and all_tokens[0].b <= ending_index:
            current_token = all_tokens.popleft()
            all_tokens_with_sentence_endings.append(current_token)
        all_tokens_with_sentence_endings.append(Token.sentence_end())
    # add the rest of tokens to the resulting list if any
    if all_tokens:
        all_tokens_with_sentence_endings.extend(list(all_tokens))
        all_tokens_with_sentence_endings.append(Token.sentence_end())

    lines: List[str] = \
        [tk.to_conll_line(fname=filename + ".txt") + "\n" for tk in all_tokens_with_sentence_endings]
    # with open(os.path.join(output_dir, filename + ".txt"), "w") as f_to_write:
    #     f_to_write.writelines(lines)

    # write all into one file
    LOCK.acquire()
    one_file.extend(lines)
    LOCK.release()


if __name__ == '__main__':
    input_dir: str
    output_dir: str
    file_name: Optional[str]
    filenames: List[str]
    input_dir, output_dir, file_name = get_dir()
    # file_name = "record-84"
    if file_name is None:
        filenames = [x[:-4] for x in os.listdir(os.path.join(input_dir, "txt")) if x.endswith(".txt")]
        if not filenames:
            sys.exit("No file found from the input dir: " + os.path.join(input_dir, "txt"))
        filenames.sort()
    else:
        filenames = [file_name]

    with open(r"/Users/chenkx/Box Sync/NLP group/2010 i2b2 challenge - rel/dev_split_threeway.txt") as f:
        wanted: List[str] = [i.strip()[:-4] for i in f.readlines()]

    print(wanted)

    output_dir = r"/Users/chenkx/Box Sync/NLP group/2010 i2b2 challenge - rel/threeway_CoNLL"

    # write all data to one test file
    one_file: List[str] = []
    LOCK = threading.Lock()

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for filename in filenames:
            futures.append(executor.submit(run, filename))
        for f in futures:
            f.result()

    # write all to one file
    # with open(os.path.join(output_dir, "conll_in_one_file.txt"), "w") as f:
    #     f.writelines([s for s in one_file])

    print(f"Found {len(filenames)} notes in: {input_dir}. \nConverted them to CONLL and saved to: \"{output_dir}\"")
