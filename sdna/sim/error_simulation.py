# -*- coding: utf-8 -*-

import re
import numpy as np
import random

from sdna.sim.core import *


class ErrorSimulation(object):

    MUTATION_MODES = ["insertion", "deletion", "mismatch"]

    def __init__(self, sequence, seed=None):
        """
        Class is used to apply modifications to a sequence.

        :param sequence: The sequence as string which will be modified.
        :param seed: Specify a integer number, this allows to reproduce the results.
        """
        self.sequence = sequence
        self.modified = set()
        self.indel_multiplier = 1
        np.random.seed(seed if seed is not None else 0)

    def _m_function(self, mode):
        """
        Returns the selected mutation mode function.

        :param mode: Which modification should be used.
        :return: Mutation function.
        """
        if mode == "insertion" or mode == "deletion":
            return self._indel
        if mode == "mismatch":
            return self._mismatch

    def _indel(self, mode, position_range=None, pattern=None):
        """
        Applies a insertion or deletion to the sequence.

        :param mode: Which modification is to be applied.
        :param position_range: Contains either the information at which position the sequence should be changed,
        or the keyword 'random' for a random position, or the keyword 'homopolymer' for a random homopolymer in the sequence.
        :param pattern: The pattern according to which the sequence is to be changed.
        """
        if position_range == "random" or position_range is None:
            sub_seq = range(0, len(self.sequence))
        else:
            if position_range == "homopolymer":
                p = Homopolymers.evaluate(self.sequence)
                if len(p) >= 1:
                    self._homopolymer(mode, p, pattern)
                    return
            sub_seq = range(position_range[0], position_range[1] + 1)

        if pattern is None:
            pos = 0
            n = 0
            while pos in self.modified:
                pos = np.random.choice(sub_seq)
                n += 1
                if n == len(sub_seq)**2:      # prevent endless loop
                    return
            self._modify_sequence(mode, pos)
        else:
            base = np.random.choice(list(pattern.keys()), p=list(pattern.values()))
            pos = 0
            n = 0
            while self.sequence[pos] != base and pos in self.modified:
                pos = np.random.choice(sub_seq)
                n += 1
                if n == len(sub_seq)**2:
                    return
            self._modify_sequence(mode, pos)

    def _homopolymer(self, mode, polymers, pattern):
        """
        Applies a insertion or deletion to the sequence where a homopolymer occurs.

        :param mode: Which modification is to be applied.
        :param polymers: List of dictionaries with the homopolymers that occur in the sequence.
        :param pattern: The pattern according to which the sequence is to be changed.
        """
        p = {e["base"] for e in polymers if e["base"] != " "}
        if not p:
            return self._indel(mode)

        if pattern:
            weights = {k: v for k, v in pattern.items() if k in p}
        else:
            weights = {e["base"]: None for e in p}
            weights.update((k, 1.0 / len(weights)) for k in weights)

        norm_weights = {k: float(v) / sum(weights.values()) for k, v in weights.items()}
        base = np.random.choice(list(weights.keys()),
                                p=list(norm_weights.values()))
        polymer = np.random.choice([e for e in polymers if e["base"] == base])

        pos = np.random.choice(range(polymer["start_pos"], polymer["end_pos"] + 1))
        n = 0
        while pos in self.modified:
            pos = np.random.choice(range(polymer["start_pos"], polymer["end_pos"] + 1))
            n += 1
            if n == (polymer["start_pos"] - polymer["end_pos"])**2:  # prevent endless loop
                return
        self._modify_sequence(mode, pos)

    def _mismatch(self, mode, position_range=None, pattern=None):
        """
        Applies a mismatch to the sequence.

        :param mode: Which modification is to be applied.
        :param position_range: Contains either the information at which position the sequence should be changed,
        or the keyword 'random' for a random position, or the keyword 'homopolymer' for a random homopolymer in the sequence.
        :param pattern: The pattern according to which the sequence is to be changed.
        """
        if position_range is None:
            sub_seq = range(0, len(self.sequence))
        else:
            sub_seq = range(position_range[0], position_range[1] + 1)

        if not pattern:
            pos = 0
            n = 0
            while pos in self.modified:
                pos = np.random.choice(sub_seq)
                n += 1
                if n == len(sub_seq)**2:      # prevent endless loop
                    return
            self._modify_sequence(mode, pos)
        else:
            reg = re.compile("|".join(pattern.keys()))
            try:
                choices = [(match.span(), match.group())
                           for match in re.finditer(reg, self.sequence[position_range[0]:position_range[1] + 1] if position_range else self.sequence)]
                e = choices[np.random.choice(len(choices))]
            except ValueError:
                return self._mismatch(mode, position_range)
            base = np.random.choice(list(pattern[e[1]].keys()),
                                    p=list(pattern[e[1]].values()))
            self.sequence = self.sequence[:e[0][0]] + base + self.sequence[e[0][1]:]

    def _modify_sequence(self, mode, position):
        """
        The sequence is modified according to the corresponding mode.

        :param mode: Which modification is to be applied to the position.
        :param position: The position of the sub sequence to be modified.
        """
        assert mode in ErrorSimulation.MUTATION_MODES
        if mode == "deletion":
            self.sequence = self.sequence[:position] + " " + self.sequence[position+1:]
            self.modified.add(position)
        elif mode == "insertion":
            base = np.random.choice(["A", "T", "C", "G"])
            self.sequence = self.sequence[:position] + base + self.sequence[position:]
            self.modified.add(position)
        else:
            base = np.random.choice(["A", "T", "C", "G"])
            self.sequence = self.sequence[:position] + base + self.sequence[position+1:]
            self.modified.add(position)

    def _random_mutation_function(self, position_range=None, modes=None, probabilities=None):
        """
        Selects a random mode of modification based on the probabilities.

        :param position_range: The range of the sub sequence to be modified.
        :param modes: Restrict which modifications should be applied to the sequence.
        :param probabilities: Probabilities for which mode should be selected.
        """
        probabilities = probabilities if not None else [0.33, 0.34, 0.33]
        mode = np.random.choice(ErrorSimulation.MUTATION_MODES, p=probabilities)
        if modes is None or mode in modes:
            self._m_function(mode)(mode=mode, position_range=position_range)

    def apply_mutations_by_detection(self, errors, modes=None):
        """
        Applies all errors to the sequence, which originate from the error detection.

        :param errors: List of dictionaries with all error rates.
        :param modes: Restrict which modifications should be applied to the sequence.
        """
        self.modified = set()       # reset all modifications
        for source in errors:
            for err in source:
                if np.random.random() <= err["error_prob"]:
                    try:
                        position_range = [err["start_pos"], err["end_pos"]]
                    except KeyError:
                        position_range = None
                    self._random_mutation_function(position_range=position_range, modes=modes)

    def apply_mutations_by_source(self, errors, modes=None):
        """
        Applies all errors to the sequence, which originate from the error sources.

        :param errors: List of dictionaries with all error rates.
        :param modes: Restrict which modifications should be applied to the sequence.
        """
        #cum_rate = 0.0
        #num_errs = 0
        print(self.indel_multiplier)
        for rate in errors:
            self.modified = set()  # reset all modifications for each progress
            for mode in ErrorSimulation.MUTATION_MODES:
                if modes is None or mode in modes:

                    #cum_rate+=rate["err_rate"]["raw_rate"]
                    err_rate = rate["err_rate"]["raw_rate"] * rate["err_rate"][mode]
                    if mode in ["insertion", "deletion"]:
                        err_rate = err_rate*self.indel_multiplier
                    err_num_list = range(round((len(self.sequence) * err_rate)))
                    #This is new to account for rare errors
                    if not err_num_list:
                        c = 0
                        for i in range(len(self.sequence)):
                            if random.random() < err_rate:
                                c += 1
                        err_num_list = range(c)

                    for n in err_num_list: #range(round((len(self.sequence) * err_rate))):
                        try:
                            position_range = np.random.choice(list(rate["err_attributes"][mode]["position"].keys()),
                                                              p=list(rate["err_attributes"][mode]["position"].values()))
                        except KeyError:
                            position_range = None
                        try:
                            pattern = rate["err_attributes"][mode]["pattern"]
                        except KeyError:
                            pattern = None
                        self._m_function(mode)(mode=mode, position_range=position_range, pattern=pattern)
                        #num_errs+=1
        #print(num_errs)
        #print(cum_rate)