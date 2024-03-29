# -*- coding: utf-8 -*-

import torch
import numpy as np
from collections import defaultdict

from sdna.sim import *

BASE_REPRESENTATION = {(-1.0, -1.0): 'A',
                       (-1.0,  1.0): 'G',
                       ( 1.0, -1.0): 'T',
                       ( 1.0,  1.0): 'C'}

NUMBER_REPRESENTATION = {'A': (-1.0, -1.0),
                         'G': (-1.0,  1.0),
                         'T': ( 1.0, -1.0),
                         'C': ( 1.0,  1.0)}

def num_to_base(pos, num):
    base_rep = {(-1.0, -1.0): 0,
                 (-1.0, 1.0): 1,
                 (1.0, -1.0): 2,
                 (1.0,  1.0): 3}

    base_map = {0: 'A', 1: 'G', 2: 'C', 3: 'T'}

    rep = base_rep[num]
    base = base_map[(rep+pos)%4]
    return base

def base_to_num(base, pos):
    base_rep = {0: (-1.0, -1.0), 1: (-1.0, 1.0), 2: (1.0, -1.0), 3: (1.0, 1.0)}
    base_map = {'A': 0, 'G': 1, 'C': 2, 'T': 3}

    num = base_map[base]
    rep = base_rep[(num-pos)%4]
    return rep

class Channel(object):
    def __init__(self, arguments):
        """
        Class is used to generate a 'noisy' channel.

        :param arguments: Arguments as dictionary.
        """
        self.args = arguments
        self._dna_simulator = Sim(self.args)

    def generate_noise(self, inputs, padding, seed, validate, channel="basic_dna"): #dna #continuous #basic_dna
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
            return self._dna_channel(inputs, padding, seed, validate)
        elif channel.lower() == "bec":      # binary erasure channel
            shape = (inputs.size()[0], inputs.size()[1] + padding, inputs.size()[2])
            sigma = 0.1
            return torch.from_numpy(np.random.choice([0.0, 1.0], shape, p=[0.1, 1 - sigma]))
        elif channel.lower() == "awgn":     # additive white gaussian noise
            shape = (inputs.size()[0], inputs.size()[1] + padding, inputs.size()[2])
            sigma = 0.1
            return sigma * torch.randn(shape, dtype=torch.float32)
        elif channel.lower() == "continuous":
            return self._continuous_channel(inputs, padding, seed, validate)
        elif channel.lower() == "basic_dna":
            return self._basic_dna_channel(inputs, padding, seed, validate)
        elif channel.lower() == "conc_dna":
            return self._conc_dna_channel(inputs, padding, seed, validate)
        else:
            return self._dna_channel(inputs, padding, seed, validate)
    def _conc_dna_channel(self, inputs, padding, seed, validate):
        p_insert = sum([(self._dna_simulator.error_rates[i]["err_rate"]["raw_rate"]
                         * self._dna_simulator.error_rates[i]["err_rate"]["insertion"])
                        for i in range(len(self._dna_simulator.error_rates))]) #0.0039
        p_delete = sum([(self._dna_simulator.error_rates[i]["err_rate"]["raw_rate"]
                         * self._dna_simulator.error_rates[i]["err_rate"]["deletion"])
                        for i in range(len(self._dna_simulator.error_rates))]) #0.0082
        p_sub = sum([(self._dna_simulator.error_rates[i]["err_rate"]["raw_rate"]
                         * self._dna_simulator.error_rates[i]["err_rate"]["mismatch"])
                        for i in range(len(self._dna_simulator.error_rates))]) #0.0238

        if not validate and np.random.randint(3) == 0:  # Account for error-free sequences
            return inputs

        if validate:
            modes = ["insertion", "deletion", "mismatch"]
        elif padding <= 0:
            modes = ["mismatch"]
            padding = 0
        else:
            modes = ["insertion", "deletion", "mismatch"]
        #padding=0
        shape = (inputs.size()[0], inputs.size()[1], inputs.size()[2])

        #outp = inputs
        x_sys = inputs[:, :, 0].view((inputs.size()[0], inputs.size()[1], 1)).cpu().detach()
        x_p1 = inputs[:, :, 1].view((inputs.size()[0], inputs.size()[1], 1)).cpu().detach()
        x_p2 = inputs[:, :, 2].view((inputs.size()[0], inputs.size()[1], 1)).cpu().detach()
        x = torch.cat((x_sys, x_p1, x_p2), dim=1)
        x_shape = (x.size()[0], x.size()[1], x.size()[2])
        x = x.cpu().detach().numpy()
        outp = list()
        for z, code in enumerate(x):
            inserts = 0
            deletions = 0
            subs = 0

            # len code divided by two, as there are always two consecutive values changed, given that one base encodes two values
            random_nums = np.random.uniform(0, 1, size=len(code)//2)
            for i in range(len(random_nums)):
                if "insertion" in modes:
                    if random_nums[i] < p_insert:
                        inserts += 1
                if "deletion" in modes:
                    if random_nums[i] < p_delete:
                        deletions += 1
                if "mismatch" in modes:
                    if random_nums[i] < p_sub:
                        subs += 1

            if subs:
                indices = np.random.choice(range(0, len(code), 2), replace=False, size=subs)
                for ind in indices:
                    code[ind] = np.random.choice([1.0, -1.0], 1)
                    code[ind+1] = np.random.choice([1.0, -1.0], 1)
            if inserts:
                for _ in range(inserts):
                    index = np.random.choice(range(0, len(code), 2))
                    val = np.array([np.random.choice([1.0, -1.0], 1), np.random.choice([1.0, -1.0], 1)])
                    code = np.insert(code, index, val, axis=0)
            if deletions:
                indices = np.random.choice(range(0, len(code), 2), deletions)
                indices_pairs = [ind + i for i in range(2) for ind in indices]
                code = np.delete(code, indices_pairs).reshape((-1, 1))
            if len(code) > x_shape[1]:
                code = code[:x_shape[1]]
            elif len(code) < x_shape[1]:
                diff = x_shape[1] - len(code)
                code = np.concatenate((code, np.zeros((diff, 1))), axis=0)
            outp.append(code)
        outp = np.array(outp).astype("float32")
        outp = torch.from_numpy(outp)
        x_out = outp.unsqueeze(-1)
        x_sys_out, x_p1_out, x_p2_out = torch.split(x_out, shape[1], dim=1)
        outp = torch.cat([x_sys_out, x_p1_out, x_p2_out], dim=2)
        return outp

    def _basic_dna_channel(self, inputs, padding, seed, validate):
        p_insert = sum([(self._dna_simulator.error_rates[i]["err_rate"]["raw_rate"]
                         * self._dna_simulator.error_rates[i]["err_rate"]["insertion"])
                        for i in range(len(self._dna_simulator.error_rates))])
        p_delete = sum([(self._dna_simulator.error_rates[i]["err_rate"]["raw_rate"]
                         * self._dna_simulator.error_rates[i]["err_rate"]["deletion"])
                        for i in range(len(self._dna_simulator.error_rates))])
        p_sub = sum([(self._dna_simulator.error_rates[i]["err_rate"]["raw_rate"]
                         * self._dna_simulator.error_rates[i]["err_rate"]["mismatch"])
                        for i in range(len(self._dna_simulator.error_rates))])

        #if not validate and np.random.randint(3) == 0:  # Account for error-free sequences
        #    return inputs

        if validate:
            modes = ["insertion", "deletion", "mismatch"]
        elif padding <= 0:
            modes = ["mismatch"]
            padding = 0
        else:
            modes = ["insertion", "deletion", "mismatch"] #remove mismatch
        #padding=0
        shape = (inputs.size()[0], inputs.size()[1], inputs.size()[2])

        x_sys = inputs[:, :, 0].view((inputs.size()[0], inputs.size()[1], 1)).cpu().detach().numpy()
        x_p1 = inputs[:, :, 1].view((inputs.size()[0], inputs.size()[1], 1)).cpu().detach().numpy()
        x_p2 = inputs[:, :, 2].view((inputs.size()[0], inputs.size()[1], 1)).cpu().detach().numpy()
        # Add insertions and deletions
        outp = list()
        for seq in [x_sys, x_p1, x_p2]:
            out_tens = list()
            for z, code in enumerate(seq):
                code = list(code)
                inserts = 0
                deletions = 0
                subs = 0

                # len code divided by two, as there are always two consecutive values changed, given that one base encodes two values
                random_nums = np.random.uniform(0, 1, size=len(code) // 2)
                for i in range(len(random_nums)):
                    if "insertion" in modes:
                        if random_nums[i] < p_insert:
                            inserts += 1
                    if "deletion" in modes:
                        if random_nums[i] < p_delete:
                            deletions += 1
                    if "mismatch" in modes:
                        if random_nums[i] < p_sub:
                            subs += 1

                if subs:
                    indices = np.random.choice(range(0, len(code), 2), replace=False, size=subs)
                    for ind in indices:
                        code[ind] = np.random.choice([1.0, -1.0], 1)
                        code[ind + 1] = np.random.choice([1.0, -1.0], 1)
                if inserts:
                    for _ in range(inserts):
                        index = np.random.choice(range(0, len(code), 2))
                        val = np.array([np.random.choice([1.0, -1.0], 1), np.random.choice([1.0, -1.0], 1)])
                        code = np.insert(code, index, val, axis=0)
                if deletions:
                    indices = np.random.choice(range(0, len(code), 2), deletions)
                    indices_pairs = [ind + i for i in range(2) for ind in indices]
                    code = np.delete(code, indices_pairs).reshape((-1, 1))
                if len(code) > shape[1]:
                    code = code[:shape[1]]
                elif len(code) < shape[1]:
                    diff = shape[1] - len(code)
                    code = np.concatenate((code, np.zeros((diff, 1))), axis=0)
                out_tens.append(code)
            outp.append(out_tens)
        outp = np.array(outp).astype("float32")
        outp = torch.from_numpy(outp)
        outp = torch.cat([outp[0], outp[1], outp[2]], dim=2)
        return outp

    def _continuous_channel(self, inputs, padding, seed, validate):
        p_insert = sum([(self._dna_simulator.error_rates[i]["err_rate"]["raw_rate"]
                         * self._dna_simulator.error_rates[i]["err_rate"]["insertion"])
                        for i in range(len(self._dna_simulator.error_rates))])
        p_delete = sum([(self._dna_simulator.error_rates[i]["err_rate"]["raw_rate"]
                         * self._dna_simulator.error_rates[i]["err_rate"]["deletion"])
                        for i in range(len(self._dna_simulator.error_rates))])
        p_sub = sum([(self._dna_simulator.error_rates[i]["err_rate"]["raw_rate"]
                         * self._dna_simulator.error_rates[i]["err_rate"]["mismatch"])
                        for i in range(len(self._dna_simulator.error_rates))])

        #if not validate or np.random.randint(3) == 0:  # Account for error-free sequences
        #    return inputs

        if validate:
            modes = ["insertion", "deletion", "mismatch"]
        elif padding <= 0:
            modes = ["mismatch"]
            padding = 0
        else:
            modes = ["insertion", "deletion", "mismatch"] #remove mismatch

        shape = (inputs.size()[0], inputs.size()[1], inputs.size()[2])

        outp = inputs
        x_sys = outp[:, :, 0].view((outp.size()[0], outp.size()[1], 1)).cpu().detach().numpy()
        x_p1 = outp[:, :, 1].view((outp.size()[0], outp.size()[1], 1)).cpu().detach().numpy()
        x_p2 = outp[:, :, 2].view((outp.size()[0], outp.size()[1], 1)).cpu().detach().numpy()
        # Add insertions and deletions
        outp = list()
        for seq in [x_sys, x_p1, x_p2]:
            out_tens = list()
            for z, code in enumerate(seq):
                code = list(code)
                inserts = 0
                deletions = 0
                subs = 0
                random_nums = np.random.uniform(0, 1, size=len(code) // 2)
                for i in range(len(random_nums)):
                    if "insertion" in modes:
                        if random_nums[i] < p_insert:
                            inserts += 1
                    if "deletion" in modes:
                        if random_nums[i] < p_delete:
                            deletions += 1
                    if "mismatch" in modes:
                        if random_nums[i] < p_sub:
                            subs += 1
                if subs:
                    indices = np.random.choice(range(0, len(code), 2), replace=False, size=subs)
                    for ind in indices:
                        code[ind] = np.random.normal(0, 0.7, 1)
                        code[ind + 1] = np.random.normal(0, 0.7, 1)
                if inserts:
                    for _ in range(inserts):
                        index = np.random.choice(range(0, len(code), 2))
                        val = np.array([np.random.normal(0, 0.7, 1), np.random.normal(0, 0.7, 1)])
                        code = np.insert(code, index, val, axis=0)
                if deletions:
                    indices = np.random.choice(range(0, len(code), 2), deletions)
                    indices_pairs = [ind + i for i in range(2) for ind in indices]
                    code = np.delete(code, indices_pairs).reshape((-1, 1))
                if len(code) > shape[1]:
                    code = code[:shape[1]]
                elif len(code) < shape[1]:
                    diff = inputs.shape[1] - len(code)
                    code = np.concatenate((code, np.zeros((diff, 1))), axis=0)
                out_tens.append(code)
            outp.append(out_tens)
        outp = np.array(outp).astype("float32")
        outp = torch.from_numpy(outp)
        outp = torch.cat([outp[0], outp[1], outp[2]], dim=2)
        return outp

        '''
        if "mismatch" in modes:
            outp = inputs + (awgn_var ** 0.5) * torch.randn(shape)
        else:
            outp = inputs
        '''
        '''
        outp = inputs
        x_sys = outp[:, :, 0].view((outp.size()[0], outp.size()[1], 1)).cpu().detach().numpy()
        x_p1 = outp[:, :, 1].view((outp.size()[0], outp.size()[1], 1)).cpu().detach().numpy()
        x_p2 = outp[:, :, 2].view((outp.size()[0], outp.size()[1], 1)).cpu().detach().numpy()
        # Add insertions and deletions
        outp = list()
        for seq in [x_sys, x_p1, x_p2]:
            out_tens = list()
            for z, code in enumerate(seq):
                code = list(code)
                inserts = 0
                deletions = 0
                subs = 0
                for i in range(len(code)):
                    if "insertion" in modes:
                        if np.random.uniform(0, 1, 1) < p_insert:
                            inserts += 1
                    if "deletion" in modes:
                        if np.random.uniform(0, 1, 1) < p_delete:
                            deletions += 1
                    if "mismatch" in modes:
                        if np.random.uniform(0, 1, 1) < p_sub/2: #p_sub has to be divided by two, as each error changes two consecutive symbols, given that 1 base encodes two symbols/bits
                            subs += 1
                if subs:
                    for _ in range(subs):
                        index = np.random.choice(range(0, len(code), 2))
                        val = [np.random.uniform(-7, 7, 1), np.random.uniform(-7, 7, 1)]
                        code = code[:index] + val + code[index+2:]
                if inserts:
                    for _ in range(inserts):
                        index = np.random.randint(low=0, high=len(code))
                        val = np.random.uniform(-7, 7, 1)
                        code = code[:index] + [np.array(val)] + (code[index:])
                if deletions:
                    for _ in range(deletions):
                        index = np.random.randint(low=0, high=len(code))
                        if index == len(code)-1:
                            code = code[:index]
                        else:
                            code = code[:index] + code[index+1:]
                if len(code) > inputs.shape[1]:
                    code = code[:inputs.shape[1]]
                elif len(code) < inputs.shape[1]:
                    diff = inputs.shape[1] - len(code)
                    code = code + [0]*diff
                out_tens.append(code)
            outp.append(out_tens)
        #if type(outp)
        outp = np.array(outp).astype("float32")
        outp = torch.from_numpy(outp)
        outp = torch.tensor(outp)
        x_sys_out = torch.tensor(outp[0]).unsqueeze(-1)
        x_p1_out = torch.tensor(outp[1]).unsqueeze(-1)
        x_p2_out = torch.tensor(outp[2]).unsqueeze(-1)
        outp = torch.cat([x_sys_out, x_p1_out, x_p2_out], dim=2)
        return outp
        '''
        '''
        if "mismatch" in modes:
            x_sys = outp[:, :, 0].view((outp.size()[0], outp.size()[1], 1)).cpu().detach().numpy()
            x_p1 = outp[:, :, 1].view((outp.size()[0], outp.size()[1], 1)).cpu().detach().numpy()
            x_p2 = outp[:, :, 2].view((outp.size()[0], outp.size()[1], 1)).cpu().detach().numpy()
            # Add insertions and deletions
            outp = list()
            for seq in [x_sys, x_p1, x_p2]:
                out_tens = list()
                for z, code in enumerate(seq):
                    code = list(code)
                    subs = 0
                    for i in range(len(code)//2):
                        if np.random.uniform(0, 1, 1) < p_sub:
                            subs += 1
                    if subs:
                        for _ in range(subs):
                            index = np.random.choice(range(0, len(code), 2))
                            val = np.array([np.random.uniform(-7, 7, 1), np.random.uniform(-7, 7, 1)])
                            code = code[:index] + val + (code[index+2:])
                    out_tens.append(code)
                outp.append(out_tens)
                            

        if "insertion" in modes or "deletion" in modes:
            x_sys = outp[:, :, 0].view((outp.size()[0], outp.size()[1], 1)).cpu().detach().numpy()
            x_p1 = outp[:, :, 1].view((outp.size()[0], outp.size()[1], 1)).cpu().detach().numpy()
            x_p2 = outp[:, :, 2].view((outp.size()[0], outp.size()[1], 1)).cpu().detach().numpy()
            # Add insertions and deletions
            outp = list()
            for seq in [x_sys, x_p1, x_p2]:
                out_tens = list()
                for z, code in enumerate(seq):
                    code = list(code)
                    inserts = 0
                    deletions = 0
                    for i in range(len(code)):
                        if np.random.uniform(0, 1, 1) < p_insert:
                            inserts += 1
                        if np.random.uniform(0, 1, 1) < p_delete:
                            deletions += 1
                    if inserts:
                        for _ in range(inserts):
                            index = np.random.randint(low=0, high=len(code))
                            val = np.random.uniform(-7, 7, 1)
                            code = code[:index] + [np.array(val)] + (code[index:])
                    if deletions:
                        for _ in range(deletions):
                            index = np.random.randint(low=0, high=len(code))
                            if index == len(code)-1:
                                code = code[:index]
                            else:
                                code = code[:index] + code[index+1:]
                    if len(code) > inputs.shape[1]:
                        code = code[:inputs.shape[1]]
                    elif len(code) < inputs.shape[1]:
                        diff = inputs.shape[1] - len(code)
                        code = code + [0]*diff
                    out_tens.append(code)
                outp.append(out_tens)
        if "insertion" in modes or "deletion" in modes:
            #if type(outp)
            outp = np.array(outp).astype("float32")
            outp = torch.from_numpy(outp)
            outp = torch.tensor(outp)
            x_sys_out = torch.tensor(outp[0]).unsqueeze(-1)
            x_p1_out = torch.tensor(outp[1]).unsqueeze(-1)
            x_p2_out = torch.tensor(outp[2]).unsqueeze(-1)
            outp = torch.cat([x_sys_out, x_p1_out, x_p2_out], dim=2)
        return outp
    '''

    def _dna_channel(self, inputs, padding, seed, validate):
        """
        The function generates noise from the encoding stream using the DNA synthesis, storage and sequencing simulator.

        :param inputs: Input tensors from encoder.
        :param padding: The number of bits by which the code should be padded.
        :param seed: Specify a integer number, this allows to reproduce the results.
        :return:  Output tensor containing noise which can be applied on input tensor.

        :note: The output tensor becomes larger depending on the padding, with a padding of zero only
        mismatches are applied.
        """
        if validate:
            modes = ["insertion", "deletion", "mismatch"]
        elif padding <= 0:
            modes = ["mismatch"]
            padding = 0
        else:
            modes = ["insertion", "deletion"]
        #padding=0
        shape = (inputs.size()[0], inputs.size()[1] + padding, inputs.size()[2])

        x_noisy = np.empty(shape, dtype=np.float32)
        for i, code in enumerate(inputs):
            x_in = code.cpu().detach().numpy()  # tensor can never be copied directly from the GPU to numpy structure
            seq_enc = Channel.bits_to_sequence(x_in, shape)        # 1. => transform bits into sequence
            p_seed = int(seed % (i + 1))    # 2. => apply noisy channel on sequence
            #if np.random.randint(3) == 0:  # Account for error-free sequences
            #    seq_dec = seq_enc
            #else:
            seq_dec = self.apply_sequence_errors(seq_enc, p_seed, modes)     # apply mutations on sequence
            x_out = Channel.sequence_to_bits(seq_dec, shape)  # 3. => transform sequence back into bits
            x_out[:inputs.size()[1], :] = x_out[:inputs.size()[1], :] * x_in
            x_noisy[i] = x_out      # 4. assign the difference between the bits
        x = torch.from_numpy(x_noisy)
        return x

    def evaluate(self, inputs, kl=None):
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
            if kl:
                error_probability += kl.calculate_kl_divergence(seq_enc)

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
        mod_runner = 0
        for j in range(shape[2]):
            x = bits[:, j]
            for n_1, n_2 in zip(x[0::2], x[1::2]):  # get sequence from code
                try:
                    seq += BASE_REPRESENTATION[n_1, n_2]
                    #seq += num_to_base(mod_runner, (n_1, n_2))
                    mod_runner+=1
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
        mod_runner = 0
        for j in range(shape[2]):
            for e_i in range(0, n):
                if e_i >= (shape[1] * 0.5) or int(j * n / shape[2]) + e_i >= n:
                    # print("WARNING: Padding is not sufficient, sequence cannot be mapped correctly into the bitstream.")
                    break
                e_1, e_2 = NUMBER_REPRESENTATION[sequence[int(j * n / shape[2]) + e_i]]
                #e_1, e_2 = base_to_num(sequence[int(j * n / shape[2]) + e_i], mod_runner)
                mod_runner+=1
                bits[(shape[1] * j) + e_i * 2] = e_1
                bits[(shape[1] * j) + e_i * 2 + 1] = e_2
        return np.reshape(bits, (shape[1], shape[2]), order='F')


class CalcLoss:
    def __init__(self, data):
        #self.data = data
        # Define a symbol-based encoding scheme
        self.symbols = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        self.max_run_length = len(self.symbols) - 1  # The maximum run length is represented by the last symbol
        self.optimal_run_length = 3 #ToDo: HARDCODED! HAS TO CHANGE!
        self.encoded_data = self.encode_data(data)
        self.loss = self.run_length_loss(data, self.encoded_data) #+ self.balance_loss(data, self.encoded_data) + self.pattern_loss(data, self.encoded_data)

    # Define a function to encode the data
    def encode_data(self, data):
        #print(data)
        encoded_data = []
        run_length = 0
        current_symbol = None
        for symbol in data:
            if symbol == current_symbol:
                run_length += 1
            else:
                if current_symbol is not None:
                    encoded_data.append(self.symbols[min(run_length, self.max_run_length)])
                current_symbol = symbol
                run_length = 1
        encoded_data.append(self.symbols[min(run_length, self.max_run_length)])  # Don't forget to append the last symbol
        #print(encoded_data)
        return encoded_data

    # Define a function to decode the data
    def decode_data(self, encoded_data):
        decoded_data = []
        for symbol in encoded_data:
            run_length = self.symbols.index(symbol) + 1
            decoded_data.extend([symbol] * run_length)
        return decoded_data

    # Define a custom loss function to penalize the encoder for using symbols that represent long runs of a particular symbol
    def run_length_loss(self, y_true, y_pred):
        loss = 0
        for yt, yp in zip(y_true, y_pred):
            #if yp in self.symbols[self.max_run_length:]:
                #loss += 1
            if yp in self.symbols[:self.optimal_run_length+1]:
                loss += 0
                #loss += ord(yp.lower())-96
            else:
                loss += (ord(yp.lower()) - 96) #/2 #*2 #ToDo: do it quadratic?
        return loss

    # Define a custom loss function to penalize the encoder for producing an encoded representation that is not weakly balanced
    def balance_loss(self, y_true, y_pred):
        zero_count = 0
        for yp in y_pred:
            if yp == 'A' or yp == "T":
                zero_count += 1
        balance = zero_count / len(y_pred)
        if balance < 0.4 or balance > 0.6:
            return 1
        return 0

    # Define a custom loss function to penalize the encoder for producing an encoded representation that contains forbidden patterns
    def pattern_loss(self, y_true, y_pred):
        loss = 0
        forbidden_patterns = []  # ['010010', '101101']
        for pattern in forbidden_patterns:
            if pattern in ''.join(y_pred):
                loss += 1
        return loss


class MarkovModelKL:
    def __init__(self, order):
        self.order = order
        self.symbol_probs = defaultdict(lambda: defaultdict(int))

    def fit(self, codeword_path):
        with open(codeword_path, "r") as fasta:
            for line in fasta.readlines():
                if not line.startswith(">"):
                    sequence = line.strip()
                    for i in range(self.order, len(sequence)):
                        preceding_sequence = sequence[i-self.order:i]
                        symbol = sequence[i]
                        self.symbol_probs[preceding_sequence][symbol] += 1
            for preceding_sequence in self.symbol_probs:
                total_count = sum(self.symbol_probs[preceding_sequence].values())
                for symbol in self.symbol_probs[preceding_sequence]:
                    self.symbol_probs[preceding_sequence][symbol] /= total_count

    def calculate_kl_divergence(self, sequence):
        kl_divergence = 0.0
        for i in range(self.order, len(sequence)):
            preceding_sequence = sequence[i-self.order:i]
            symbol = sequence[i]
            if preceding_sequence not in self.symbol_probs:
                continue
            encoded_probs = torch.zeros(len(self.symbol_probs[preceding_sequence])) + 1e-9
            if symbol in self.symbol_probs[preceding_sequence]:
                encoded_probs[list(self.symbol_probs[preceding_sequence].keys()).index(symbol)] = 1.0 - (len(self.symbol_probs[preceding_sequence]) * 1e-9)
            empirical_probs = torch.tensor(list(self.symbol_probs[preceding_sequence].values()))
            kl_divergence += torch.sum(empirical_probs * (torch.log(empirical_probs) - torch.log(encoded_probs)))
        kl_divergence /= len(sequence)
        return kl_divergence if kl_divergence != float('inf') else torch.tensor(0.0)
