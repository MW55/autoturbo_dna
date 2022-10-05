# -*- coding: utf-8 -*-

import torch
import numpy as np

from sdna.sim import *

BASE_REPRESENTATION = {(-1.0, -1.0): 'A',
                       (-1.0,  1.0): 'G',
                       ( 1.0, -1.0): 'T',
                       ( 1.0,  1.0): 'C'}

NUMBER_REPRESENTATION = {'A': (-1.0, -1.0),
                         'G': (-1.0,  1.0),
                         'T': ( 1.0, -1.0),
                         'C': ( 1.0,  1.0)}


class Channel(object):
    def __init__(self, arguments):
        """
        Class is used to generate a 'noisy' channel.

        :param arguments: Arguments as dictionary.
        """
        self.args = arguments
        self._dna_simulator = Sim(self.args)

    def generate_noise(self, inputs, padding, seed, channel="dna"):
        """
        Generates a 'noisy' channel depending on the specifications.

        :param inputs: Input tensors from encoder.
        :param padding: The number of bits by which the code should be padded.
        :param seed: Specify a integer number, this allows to reproduce the results.
        :param channel: Which channel is to be used.
        :return: Output tensor containing noise which can be applied on a tensor.

        :note: The channel should actually only simulate DNA errors, the others are purely suitable for testing.
        """
        if channel.lower() == "dna":
            return self._dna_channel(inputs, padding, seed)
        elif channel.lower() == "bec":      # binary erasure channel
            shape = (inputs.size()[0], inputs.size()[1] + padding, inputs.size()[2])
            sigma = 0.1
            return torch.from_numpy(np.random.choice([0.0, 1.0], shape, p=[0.1, 1 - sigma]))
        elif channel.lower() == "awgn":     # additive white gaussian noise
            shape = (inputs.size()[0], inputs.size()[1] + padding, inputs.size()[2])
            sigma = 0.1
            return sigma * torch.randn(shape, dtype=torch.float32)
        else:
            return self._dna_channel(inputs, padding, seed)

    def _dna_channel(self, inputs, padding, seed):
        """
        The function generates noise from the encoding stream using the DNA synthesis, storage and sequencing simulator.

        :param inputs: Input tensors from encoder.
        :param padding: The number of bits by which the code should be padded.
        :param seed: Specify a integer number, this allows to reproduce the results.
        :return:  Output tensor containing noise which can be applied on input tensor.

        :note: The output tensor becomes larger depending on the padding, with a padding of zero only
        mismatches are applied.
        """
        modes = ["insertion", "deletion", "mismatch"]
        if padding <= 0:
            modes = ["mismatch"]
            padding = 0
        shape = (inputs.size()[0], inputs.size()[1] + padding, inputs.size()[2])

        x_noisy = np.empty(shape, dtype=np.float32)
        for i, code in enumerate(inputs):
            x_in = code.cpu().detach().numpy()  # tensor can never be copied directly from the GPU to numpy structure
            seq_enc = Channel.bits_to_sequence(x_in, shape)        # 1. => transform bits into sequence
            p_seed = int(seed % (i + 1))    # 2. => apply noisy channel on sequence
            if np.random.randint(3) == 0:  # Account for error-free sequences
                seq_dec = seq_enc
            else:
                seq_dec = self.apply_sequence_errors(seq_enc, p_seed, modes)     # apply mutations on sequence
            x_out = Channel.sequence_to_bits(seq_dec, shape)  # 3. => transform sequence back into bits
            x_out[:inputs.size()[1], :] = x_out[:inputs.size()[1], :] * x_in
            x_noisy[i] = x_out      # 4. assign the difference between the bits
        x = torch.from_numpy(x_noisy)
        return x

    def evaluate(self, inputs):
        """
        Determines how susceptible the given input is to DNA errors.

        :param inputs: Input tensors from encoder.
        :return: The mean DNA error probability for the given input.
        """
        shape = (inputs.size()[0], inputs.size()[1], inputs.size()[2])

        error_probability = 0.0
        for i, code in enumerate(inputs):
            x_in = code.cpu().detach().numpy()  # tensor can never be copied directly from the GPU to numpy structure
            seq_enc = Channel.bits_to_sequence(x_in, shape)  # 1. => transform bits into sequence
            error_probability += self._dna_simulator.apply_detection(seq_enc)

        error_probability /= (shape[0] * shape[2])
        return error_probability

    def apply_sequence_errors(self, sequence, seed, modes):
        """
        Calls the predefined simulator and applies sequence error to it.

        :param sequence: The sequence as string which will be modified.
        :param seed: Specify a integer number, this allows to reproduce the results.
        :param modes: Restrict which modifications should be applied to the sequence.
        :returns: Modified sequence as string.
        """
        return self._dna_simulator.apply_errors(sequence, seed, modes).replace(' ', '')

    @staticmethod
    def bits_to_sequence(bits, shape):
        """
        Transforms given bits into a DNA sequence.

        :param bits: Bits as numpy array.
        :param shape: Shape of the tensor to be handled.
        :return: DNA sequence as string.
        """
        seq = ""
        for j in range(shape[2]):
            x = bits[:, j]
            for n_1, n_2 in zip(x[0::2], x[1::2]):  # get sequence from code
                try:
                    seq += BASE_REPRESENTATION[n_1, n_2]
                except KeyError:
                    # print("WARNING: Error mapping bitstream to sequence, encoder output contains invalid bit.")
                    continue
        return seq

    @staticmethod
    def sequence_to_bits(sequence, shape):
        """
        Transforms given a DNA sequence into bits.

        :param sequence: DNA sequence as string.
        :param shape: Shape of the tensor to be handled.
        :return: Bits as numpy array.
        """
        bits = np.zeros(shape[1] * shape[2], dtype=np.float32)
        n = len(sequence)
        for j in range(shape[2]):
            for e_i in range(0, n):
                if e_i >= (shape[1] * 0.5) or int(j * n / shape[2]) + e_i >= n:
                    # print("WARNING: Padding is not sufficient, sequence cannot be mapped correctly into the bitstream.")
                    break
                e_1, e_2 = NUMBER_REPRESENTATION[sequence[int(j * n / shape[2]) + e_i]]
                bits[(shape[1] * j) + e_i * 2] = e_1
                bits[(shape[1] * j) + e_i * 2 + 1] = e_2
        return np.reshape(bits, (shape[1], shape[2]), order='F')
