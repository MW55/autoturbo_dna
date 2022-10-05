# -*- coding: utf-8 -*-

import json
import numpy as np

from scipy.interpolate import interp1d

ERROR_PROBABILITIES_JSON = "config/error_detection/probabilities.json"


class ErrorDetection(object):
    def __init__(self, config_file=ERROR_PROBABILITIES_JSON):
        """
        Class is used for the creation of a error probability function for a specific error source.

        :param config_file: Path to the json file that contains the meta data for the error probabilities.

        :note: By default the predefined json is used, this json file can be modified or exchanged.
        """
        if config_file is None:
            config_file = ERROR_PROBABILITIES_JSON
        with open(config_file) as json_file:
            ErrorDetection.config = json.load(json_file)

    config = {}

    @staticmethod
    def get_function_by_name(error_source):
        """
        Creates with the help of specified data the error probability function for a error source.

        :param error_source: Name of the error source for which the function is to be created.
        :return: Error probability function.
        """
        config = {}
        if ErrorDetection.config is None:
            return None
        if len(ErrorDetection.config) >= 1:
            config = ErrorDetection.config[error_source]

        max_x = config["maxX"]
        max_y = config["maxY"]
        data = ErrorDetection._remove_duplicates(config["data"], config["xRound"])

        x_list = np.asarray([round(e["x"], config["xRound"]) for e in data], dtype=np.float32)
        y_list = np.asarray([round(e["y"], config["yRound"]) for e in data], dtype=np.float32)

        if config["interpolation"]:
            f = ErrorDetection._interpolant(x_list, y_list)
        else:
            f = interp1d(x_list, y_list)

        def func(x):
            if x < 0.0:
                x = 0
            elif x > x_list[-1]:
                x = x_list[-1]

            res = f(x).item(0)
            if res < 0.0:
                res = 0.0
            elif res > max_y:
                res = max_y
            return 1.0 * res / 100.0

        return func

    @staticmethod
    def _remove_duplicates(data, x_round):
        """
        Removes duplicate entries in the data dictionary.

        :param data: Dictionary in which the duplicates are to be removed.
        :param x_round: Maximum value of the x in data.
        :return: Cleaned data dictionary.
        """
        res = []
        x_set = set()
        for e in data:
            x_val = e["x"]
            if round(x_val, x_round) not in x_set:
                res.append(e)
                x_set.add(x_val)
        return res

    @staticmethod
    def _interpolant(x_list, y_list):
        """
        Returns the interpolation function that maps the data.

        :param x_list: X values as numpy array.
        :param y_list: Y values as numpy array.
        :return: Interpolation function.
        """
        length = len(x_list)
        if length != len(y_list) or length == 0:
            def func(x):
                return 0

            return func
        if length == 1:
            res = +y_list[0]

            def func(x):
                return res

            return func

        indices = []    # sort x_list
        for i in range(0, length):
            indices.append(i)
        for i in range(len(indices)):   # traverse through all array elements
            min_idx = i
            for j in range(i + 1, len(indices)):
                if x_list[min_idx] > x_list[j]:
                    min_idx = j
            indices[i], indices[min_idx] = indices[min_idx], indices[i]
        old_xs = x_list
        old_ys = y_list
        xs = []
        ys = []
        for i in range(0, length):
            xs.append(+old_xs[indices[i]])
            ys.append(+old_ys[indices[i]])

        dys = []    # consecutive differences and slopes
        dxs = []
        ms = []
        for i in range(0, length - 1):
            dx = xs[i + 1] - xs[i]
            dy = ys[i + 1] - ys[i]
            dxs.append(dx)
            dys.append(dy)
            ms.append(dy / dx)

        c1s = [ms[0]]       # deg -1 coefficients
        for i in range(0, len(dxs) - 1):
            m = ms[i]
            m_next = ms[i + 1]
            if m * m_next <= 0:
                c1s.append(0)
            else:
                dx_ = dxs[i]
                dx_next = dxs[i + 1]
                common = dx_ + dx_next
                c1s.append(3 * common / ((common + dx_next) / m + (common + dx_) / m_next))
        c1s.append(ms[len(ms) - 1])

        c2s = []    # deg -2/-3 coefficients
        c3s = []
        for i in range(0, len(c1s) - 1):
            c1 = c1s[i]
            m_ = ms[i]
            inv_dx = 1 / dxs[i]
            common_ = c1 + c1s[i + 1] - m_ - m_
            c2s.append((m_ - c1 - common_) * inv_dx)
            c3s.append(common_ * inv_dx * inv_dx)

        def func(x):    # interpolant function
            index = len(xs) - 1
            if x == xs[index]:
                return ys[index]
            low = 0
            high = len(c3s) - 1
            while low <= high:
                mid = int(np.floor(0.5 * (low + high)))
                x_here = xs[mid]
                if x_here < x:
                    low = mid + 1
                elif x_here > x:
                    high = mid - 1
                else:
                    return ys[mid]

            index = max(0, high)
            diff = x - xs[index]
            diff_sq = diff * diff
            return ys[index] + c1s[index] * diff + c2s[index] * diff_sq + c3s[index] * diff * diff_sq

        return func
