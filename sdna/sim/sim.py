# -*- coding: utf-8 -*-

from sdna.sim.error_simulation import *
from sdna.sim.error_detection import *
from sdna.sim.core import *


class Sim(object):
    def __init__(self, arguments):
        """
        Class is used to simulate and detect potential errors in DNA sequences.

        :param arguments: Arguments as dictionary.
        """
        self.args = arguments
        self._prepare_error_detection()
        self._prepare_error_simulation()

    def _prepare_error_detection(self):
        """
        Loads the data for the error detection function and generates the functions from it.
        """
        detection = ErrorDetection(self.args["probabilities_json"])
        self.error_functions = {"gc_content": detection.get_function_by_name("gc_content"),
                                "kmers": detection.get_function_by_name("kmers"),
                                "homopolymers": detection.get_function_by_name("homopolymers")}

    def _prepare_error_simulation(self):
        """
        Loads the data for the error rates and prepares the data from it.
        """
        self.error_rates = []
        if self.args["synthesis"][0] != "0":
            syn_json = self.args["synthesis"][1] if len(self.args["synthesis"]) > 1 else None
            syn = ErrorSource(process="synthesis",
                              config_file=syn_json).get_by_id(self.args["synthesis"][0], multiplier=self.args["amplifier"])
            self.error_rates.append(syn)
        if self.args["storage"][0] != "0":
            sto_json = self.args["storage"][1] if len(self.args["storage"]) > 1 else None
            sto = ErrorSource(process="storage",
                              config_file=sto_json).get_by_id(self.args["storage"][0], multiplier=self.args["storage_months"] * self.args["amplifier"])
            self.error_rates.append(sto)
        if self.args["pcr"][0] != "0":
            pcr_json = self.args["pcr"][1] if len(self.args["pcr"]) > 1 else None
            pcr = ErrorSource(process="storage",
                              config_file=pcr_json).get_by_id(self.args["pcr"][0], multiplier=self.args["pcr_cycles"] * self.args["amplifier"])
            self.error_rates.append(pcr)
        if self.args["sequencing"][0] != "0":
            seq_json = self.args["sequencing"][1] if len(self.args["sequencing"]) > 1 else None
            seq = ErrorSource(process="sequencing",
                              config_file=seq_json).get_by_id(self.args["sequencing"][0], multiplier=self.args["amplifier"])
            self.error_rates.append(seq)

    def apply_errors(self, sequence, seed=None, modes=None):
        """
        Simulates potential errors that can occur during DNA synthesis, storage and sequencing.

        :param sequence: The sequence as string which will be modified.
        :param seed: Specify a integer number, this allows to reproduce the results.
        :param modes: Restrict which modifications should be applied to the sequence.
        :returns: Modified sequence as string.
        """
        seq_err = ErrorSimulation(sequence, seed=seed)
        seq_err.apply_mutations_by_source(self.error_rates, modes)
        return seq_err.sequence

    def apply_detection(self, sequence):
        """
        Detects potential errors that can occur during DNA synthesis, storage and sequencing.

        :param sequence: The sequence as string which will be modified.
        :returns: The error probability for the given sequence.
        """
        useq = UndesiredSequences(self.args["useq_json"]).evaluate(sequence)
        gc = GCContent().evaluate(sequence, window_size=self.args["gc_window"], error_function=self.error_functions["gc_content"])
        kmer = Kmers().evaluate(sequence, k=self.args["kmer_window"], error_function=self.error_functions["kmers"])
        poly = Homopolymers().evaluate(sequence, error_function=self.error_functions["homopolymers"])

        sources = [*useq, *gc, *kmer, *poly]
        error_probability = 0.0
        for error in sources:
            strength = (error["end_pos"] - error["start_pos"] + 1) / len(sequence)
            error_probability += (strength * error["error_prob"])
            if error_probability >= 1.0:
                error_probability = 1.0
                break
        return error_probability

