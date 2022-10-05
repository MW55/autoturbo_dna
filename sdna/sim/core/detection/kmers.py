# -*- coding: utf-8 -*-

class Kmers(object):
    """
    Class is primarily used to determine the error probabilities based on k-mers.
    """

    @staticmethod
    def _default_error_function(kmer_amount):
        """
        Default/Fallback function to calculate error probabilities based on k-mers.

        :param kmer_amount: Number of k-mers.
        :return: Error probability based on the amount of k-mers.
        """
        return kmer_amount ** 2 * 0.000002

    @staticmethod
    def evaluate(sequence, k=20, error_function=None):
        """
        Calculates the error probabilities based on the amount of k-mers and the window size.

        :param sequence: The sequence as string to be checked.
        :param k: Length of substrings.
        :param error_function: Error probability function.
        :return: Returns a list of dictionaries with all error probabilities.
        """
        if error_function is None:
            error_function = Kmers._default_error_function

        kmers = dict()
        for index in range(len(sequence) - k + 1):
            kmer = sequence[index:index+k]
            if kmer not in kmers:
                kmers[kmer] = [index]
            else:
                kmers[kmer].append(index)

        res = []
        counter = 0
        for kmer, indices in kmers.items():
            error_prob = error_function(len(indices))
            if error_prob > 0.0:
                for index in indices:
                    res.append({"start_pos": index,
                                "end_pos": index + k - 1,
                                "error_prob": error_prob,
                                "identifier": "kmer_" + str(counter),
                                "kmer": kmer})
                    counter += 1
        return res
