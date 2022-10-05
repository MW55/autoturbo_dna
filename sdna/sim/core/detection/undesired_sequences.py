# -*- coding: utf-8 -*-

import json
import regex as re

UNDESIRED_SEQUENCES_JSON = "../config/error_detection/undesired_sequences.json"


class UndesiredSequences(object):
    def __init__(self, seq_file=UNDESIRED_SEQUENCES_JSON):
        """
        Class is used for checking existing undesired subsequences.

        :param seq_file: Path to the json file containing well-known undesired subsequences.

        :note: By default the predefined json is used, this json file can be modified or exchanged.
        """
        if seq_file is None:
            seq_file = UNDESIRED_SEQUENCES_JSON
        with open(seq_file) as json_file:
            UndesiredSequences.sequences = json.load(json_file)

    sequences = []

    @staticmethod
    def evaluate(sequence):
        """
        Checks the sequence for undesired subsequences.

        :param sequence: The sequence as string to be checked.
        :return: Returns a list of dictionaries with all matches.
        """
        sequences = {}
        if UndesiredSequences.sequences is None:
            return []
        if len(UndesiredSequences.sequences) >= 1:
            for u_seq in UndesiredSequences.sequences:
                sequences[u_seq["sequence"]] = [float(u_seq["error_prob"]) / 100.0, u_seq["description"]]

        res = []
        counter = 0
        for m in re.finditer("|".join(sequences.keys()), sequence, overlapped=True):
            res.append({"start_pos": m.start(),
                        "end_pos": m.start() + len(m.group(0)) - 1,
                        "error_prob": sequences[m.group(0)][0],
                        "identifier": "subsequences_" + str(counter),
                        "undesired_sequence": m.group(0),
                        "description": sequences[m.group(0)][1]})
            counter += 1
        return res
