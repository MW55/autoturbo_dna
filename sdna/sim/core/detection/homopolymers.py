# -*- coding: utf-8 -*-

class Homopolymers(object):
    """
    Class is primarily used to determine the error probabilities based on homopolymers.
    """

    @staticmethod
    def _default_error_function(homopolymer_length):
        """
        Default/Fallback function to calculate error probabilities based on homopolymers.

        :param homopolymer_length: Length of homopolymer.
        :return: Error probability based on the length of homopolymers.
        """
        if homopolymer_length <= 2:
            return 0
        elif homopolymer_length <= 3:
            return 0.3
        elif homopolymer_length <= 4:
            return 0.6
        elif homopolymer_length <= 5:
            return 0.9
        else:
            return 1.0

    @staticmethod
    def evaluate(sequence, error_function=None):
        """
        Calculates the error probabilities based on the length of homopolymers.

        :param sequence: The sequence as string to be checked.
        :param error_function: Error probability function.
        :return: Returns a list of dictionaries with all error probabilities.
        """
        if error_function is None:
            error_function = Homopolymers._default_error_function

        res = []
        length = len(sequence)
        counter = 0
        base = sequence[0]
        start_pos = 0
        for index in range(1, length + 1):
            if index == length or base != sequence[index]:
                error_prob = error_function(index - start_pos)
                if error_prob > 0.0:
                    res.append({"start_pos": start_pos,
                                "end_pos": index - 1,
                                "error_prob": error_prob,
                                "identifier": "homopolymer_" + str(counter),
                                "base": base})
                    counter += 1
                start_pos = index
                base = sequence[index] if index != length else ''
        return res
