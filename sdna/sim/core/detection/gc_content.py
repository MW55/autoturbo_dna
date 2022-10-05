# -*- coding: utf-8 -*-

import math


class GCContent(object):
    """
    Class is primarily used to determine the error probabilities based on the GC-content.
    """

    @staticmethod
    def _default_error_function(gc_percentage):
        """
        Default/Fallback function to calculate error probabilities based on the GC-content.

        :param gc_percentage: The percentage of GC-content.
        :return: Error probability based on the percentage of GC-content.
        """
        gc_percentage = 1.0 * gc_percentage / 100.0
        if 0.5 <= gc_percentage <= 0.6:
            return 0
        elif gc_percentage > 0.6:
            return (gc_percentage - 0.6) * 2.5
        else:
            return (0.5 - gc_percentage) * 2

    @staticmethod
    def evaluate(sequence, window_size=15, error_function=None):
        """
        Calculates the error probabilities based on the GC-content and the window size.

        :param sequence: The sequence as string to be checked.
        :param window_size: Size of the window to be used.
        :param error_function: Error probability function.
        :return: Returns a list of dictionaries with all error probabilities.
        """
        if error_function is None:
            error_function = GCContent._default_error_function

        if len(sequence) != window_size:
            return GCContent._windowed(sequence, window_size, error_function)
        else:
            return GCContent._overall(sequence, error_function)

    @staticmethod
    def _windowed(sequence, window_size, error_function):
        """
        Calculates the error probabilities based on the GC-content for subsequences.

        :note: See function evaluate().
        """
        res = []
        length = len(sequence)
        windows = math.ceil(1.0 * length / window_size)
        counter = 0
        for i in range(windows):
            base_counter = dict()
            window_sequence = sequence[i * window_size:min((i + 1) * window_size, length)]
            window_length = len(window_sequence)
            for index in range(window_length):
                if window_sequence[index] in base_counter:
                    base_counter[window_sequence[index]] += 1
                else:
                    base_counter[window_sequence[index]] = 1
            gc_sum = 1.0 * ((base_counter["G"] if "G" in base_counter else 0.0) + (base_counter["C"] if "C" in base_counter else 0.0))
            error_prob = error_function(gc_sum / window_length * 100.0)

            if error_prob > 0.0:
                res.append({"start_pos": i * window_size,
                            "end_pos": min((i + 1) * window_size - 1, length - 1),
                            "error_prob": error_prob,
                            "identifier": "window_gc_content_" + str(counter)})
                counter += 1
        return res

    @staticmethod
    def _overall(sequence, error_function):
        """
        Calculates the error probability based on the GC-content for the whole sequence.

        :note: See function evaluate().
        """
        res = []
        length = len(sequence)
        base_counter = dict()
        for index in range(length):
            if sequence[index] in base_counter:
                base_counter[sequence[index]] += 1
            else:
                base_counter[sequence[index]] = 1
        gc_sum = 1.0 * (base_counter["G"] if "G" in base_counter else 0.0 + base_counter["C"] if "C" in base_counter else 0.0)
        error_prob = error_function(gc_sum / length * 100.0)

        res.append({"start_pos": 0,
                    "end_pos": length,
                    "error_prob": error_prob,
                    "identifier": "overall_gc_content_0"})
        return res
