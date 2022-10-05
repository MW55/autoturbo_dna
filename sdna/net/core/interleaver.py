# -*- coding: utf-8 -*-

import torch


class Interleaver(torch.nn.Module):
    def __init__(self, array=None):
        """
        Class that represents an interleaver.

        :param array: That array that is needed is used to change the order.
        """
        super(Interleaver, self).__init__()
        if array is not None:
            self.order = torch.LongTensor(array).view(len(array))

    def set_order(self, array):
        """
        Changes the array that is needed to change the order

        :param array: That array that is needed is used to change the order.
        """
        self.order = torch.LongTensor(array).view(len(array))

    def forward(self, inputs):
        """
        Calculates output tensors from input tensors based on the process.

        :param inputs: Input tensors.
        :return: Interleaved output tensor.
        """
        try:
            inputs = inputs.permute(1, 0, 2)
            x = inputs[self.order]
            x = x.permute(1, 0, 2)
            return x
        except torch.nn.modules.module.ModuleAttributeError:
            return inputs


class DeInterleaver(torch.nn.Module):
    def __init__(self, array=None):
        """
        Class that represents an de-interleaver.

        :param array: That array that is needed is used to restore order.
        """
        super(DeInterleaver, self).__init__()
        if array is not None:
            self.order = [0 for _ in range(len(array))]
            for i in range(len(array)):     # reverse array
                self.order[array[i]] = i
            self.order = torch.LongTensor(self.order).view(len(array))

    def set_order(self, array):
        """
        Changes the array that is needed to change the order

        :param array: That array that is needed is used to restore order.
        """
        self.order = [0 for _ in range(len(array))]
        for i in range(len(array)):     # reverse array
            self.order[array[i]] = i
        self.order = torch.LongTensor(self.order).view(len(array))

    def forward(self, inputs):
        """
        Calculates output tensors from input tensors based on the process.

        :param inputs: Input tensors.
        :return: De-Interleaved output tensor.
        """
        try:
            inputs = inputs.permute(1, 0, 2)
            x = inputs[self.order]
            x = x.permute(1, 0, 2)
            return x
        except torch.nn.modules.module.ModuleAttributeError:
            return inputs
