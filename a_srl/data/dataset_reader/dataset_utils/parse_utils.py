import os
import sys
import argparse

from time import gmtime, strftime

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.join(__file__, "../../../../"))))

from typing import List, Dict, Tuple, Iterable

from allennlp.data.dataset_readers.dataset_utils.ontonotes import Ontonotes, OntonotesSentence

def ontonotes_subset(ontonotes_reader: Ontonotes,
                      file_path: str,
                      domain_identifier: str) -> Iterable[OntonotesSentence]:
    """
    Iterates over the Ontonotes 5.0 dataset using an optional domain identifier.
    If the domain identifier is present, only examples which contain the domain
    identifier in the file path are yielded.
    """
    count_missing = 0
    files_to_parses = {}

    for conll_file in ontonotes_reader.dataset_path_iterator(file_path):

        if domain_identifier is None or f"/{domain_identifier}/" in conll_file:

            path = "/".join(conll_file.split("/")[-4:])

            if path not in files_to_parses:
                files_to_parses[path] = []

            for sentence in ontonotes_reader.sentence_iterator(conll_file):
                parse = sentence.parse_tree
                if not parse:
                    count_missing +=1
                else:
                    files_to_parses[path].append(parse)

    print("No. of parse trees missing in split: %d" % (count_missing))

    return files_to_parses

def write_parse_trees_to_file(ontonotes_dir: str, serialization_dir: str):

    splits = ["train", "development", "test", "conll-2012-test"]

    for split in splits:

        print("Split: ", split)
        print("Start time: ", strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))

        output_file_path = os.path.join(serialization_dir, split)

        if not os.path.isdir(output_file_path):
            os.makedirs(output_file_path)

        file_path = os.path.join(ontonotes_dir, "conll-formatted-ontonotes-5.0", "data", split)

        ontonotes_reader = Ontonotes()
        parses = ontonotes_subset(ontonotes_reader, file_path, None)

        for path in parses:

            # print(path)
            parent_dir_path = '/'.join(path.split('/')[:-1])
            if not os.path.isdir(os.path.join(output_file_path, parent_dir_path)):
                os.makedirs(os.path.join(output_file_path, parent_dir_path))

            parse_file_path = open(os.path.join(output_file_path, path), 'w')
            for parse in parses[path]:
                # parse_file_path.write(parse.pformat() + '\n')
                parse_file_path.write(parse._pformat_flat(nodesep='', parens='()', quotes=False) + '\n')
            parse_file_path.close()

        print("End time: ", strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--ontonotes_dir', type=str, help='The path to OntoNotes v.5.0 directory.')
    parser.add_argument('--serialization_dir', type=str, help='The path to save the parse trees from OntoNotes v.5.0 corpus.')

    args = parser.parse_args()

    write_parse_trees_to_file(args.ontonotes_dir, args.serialization_dir)

