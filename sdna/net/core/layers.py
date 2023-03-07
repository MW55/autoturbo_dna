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
        elif self._actf == "leakyrelu":
            return func.leaky_relu(inputs)
        elif self._actf == "gelu":
            return func.gelu(inputs)
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

class Conv1d_inc_kernel(torch.nn.Module):
    def __init__(self, actf, layers, in_channels, out_channels, kernel_size):
        """
        Applies multiple 1D convolution over an input tensor composed of several input planes.

        :param actf: The activating function to be applied.
        :param layers: The number of convolution to be used.
        :param in_channels: Number of units in the input.
        :param out_channels: Number of output channel produced by the convolution.
        :param kernel_size: Size of the convolving kernel.
        """
        super(Conv1d_inc_kernel, self).__init__()
        self._actf = actf
        self._layers = layers

        self._cnns = torch.nn.ModuleList()

        for i in range(self._layers):
            if i == 0:
                self._cnns.append(torch.nn.Conv1d(in_channels=in_channels,
                                                  out_channels=out_channels,
                                                  kernel_size=(kernel_size//2),
                                                  stride=1,
                                                  padding=(kernel_size//3),
                                                  dilation=1,
                                                  groups=1,
                                                  bias=True))
            elif i < self._layers//2:
                self._cnns.append(torch.nn.Conv1d(in_channels=out_channels,
                                                  out_channels=out_channels,
                                                  kernel_size=(kernel_size - 1) // 2,
                                                  stride=1,
                                                  padding=(kernel_size//4),
                                                  dilation=1,
                                                  groups=1,
                                                  bias=True))
            else:
                self._cnns.append(torch.nn.Conv1d(in_channels=out_channels,
                                                  out_channels=out_channels,
                                                  kernel_size=kernel_size,
                                                  stride=1,
                                                  padding=(kernel_size - 1) // 2,
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
        elif self._actf == "leakyrelu":
            return func.leaky_relu(inputs)
        elif self._actf == "gelu":
            return func.gelu(inputs)
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



class ForbiddenSeqActivation(torch.nn.Module):
    def __init__(self, window_size=8):
        super(ForbiddenSeqActivation, self).__init__()
        # Define the forbidden sequences
        self.forbidden_seqs = [
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [-1, 1, -1, 1, -1, 1, -1, 1],
            [1, -1, 1, -1, 1, -1, 1, -1],
            [1, 1, 1, 1, 1, 1, 1, 1]
        ]
        self.window_size = window_size

    def forward(self, x):
        # Convert the input tensor to a list of sequences
        input_seqs = x.tolist()
        # Iterate over each sequence and apply sliding window
        for i, seq in enumerate(input_seqs):
            for j in range(len(seq) - self.window_size + 1):
                window = seq[j:j+self.window_size]
                for forbidden_seq in self.forbidden_seqs:
                    if window == forbidden_seq:
                        # Replace the last value in the forbidden subsequence with its opposite sign
                        last_val = window[-1]
                        new_last_val = -last_val
                        input_seqs[i][j+self.window_size-1] = new_last_val
        # Convert the modified input back to a tensor and apply the ReLU activation function
        output_tensor = torch.tensor(input_seqs)
        output_tensor = torch.relu(output_tensor)
        return output_tensor


