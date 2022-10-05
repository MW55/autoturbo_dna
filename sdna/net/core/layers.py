# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as func


class Quantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        """
        Performs the forward operation.

        :param ctx: Autograd object with helper functions.
        :param inputs: Input tensors.
        :return: Output tensors.
        """
        ctx.save_for_backward(inputs)   # save tensors for backward function

        x_norm = torch.clamp(inputs, -1.0, 1.0)
        x = torch.sign(x_norm)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        """
        Applies gradient formula.

        :param ctx: Autograd object with helper functions.
        :param grad_output: Input tensors.
        :return: Output tensors.
        """
        inputs, = ctx.saved_tensors     # get saved tensors from forward function

        grad_output[inputs > 1.0] = 0
        grad_output[inputs < -1.0] = 0
        grad_output = torch.clamp(grad_output, -0.01, 0.01)
        x = grad_output.clone()
        return x, None


class Conv1d(torch.nn.Module):
    def __init__(self, actf, layers, in_channels, out_channels, kernel_size):
        """
        Applies multiple 1D convolution over an input tensor composed of several input planes.

        :param actf: The activating function to be applied.
        :param layers: The number of convolution to be used.
        :param in_channels: Number of units in the input.
        :param out_channels: Number of output channel produced by the convolution.
        :param kernel_size: Size of the convolving kernel.
        """
        super(Conv1d, self).__init__()
        self._actf = actf
        self._layers = layers

        self._cnns = torch.nn.ModuleList()

        for i in range(self._layers):
            if i == 0:
                self._cnns.append(torch.nn.Conv1d(in_channels=in_channels,
                                                  out_channels=out_channels,
                                                  kernel_size=kernel_size,
                                                  stride=1,
                                                  padding=(kernel_size // 2),
                                                  dilation=1,
                                                  groups=1,
                                                  bias=True))
            else:
                self._cnns.append(torch.nn.Conv1d(in_channels=out_channels,
                                                  out_channels=out_channels,
                                                  kernel_size=kernel_size,
                                                  stride=1,
                                                  padding=(kernel_size // 2),
                                                  dilation=1,
                                                  groups=1,
                                                  bias=True))

    def actf(self, inputs):
        """
        Activation functions which is called from forward function.

        :param inputs: Input tensors.
        :return: Output tensor to which the activation function is applied.
        """
        if self._actf == "tanh":
            return torch.tanh(inputs)
        if self._actf == "elu":
            return func.elu(inputs)
        elif self._actf == "relu":
            return torch.relu(inputs)
        elif self._actf == "selu":
            return func.selu(inputs)
        elif self._actf == "sigmoid":
            return torch.sigmoid(inputs)
        elif self._actf == "identity":
            return inputs
        else:
            return inputs

    def forward(self, inputs):
        """
        Calculates output tensors from input tensors based on the process.

        :param inputs: Input tensor.
        :return: Output tensor of encoder.
        """
        inputs = torch.transpose(inputs, 1, 2)
        x_t = inputs
        for i in range(self._layers):
            x_t = self.actf(self._cnns[i](x_t))

        x = torch.transpose(x_t, 1, 2)
        return x
