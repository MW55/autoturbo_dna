# -*- coding: utf-8 -*-
import torch

from sdna.net.core.layers import *
from sdna.net.core.interleaver import *
import numpy as np

class CoderBase(torch.nn.Module):
    def __init__(self, arguments):
        """
        Class serves as the coder template, it provides utility functions.

        :param arguments: Arguments as dictionary.
        """
        super(CoderBase, self).__init__()
        self.args = arguments

    def set_parallel(self):
        """
        Inheritance function to set the model parallel.
        """
        pass

    def set_interleaver_order(self, array):
        """
        Inheritance function to set the models interleaver/de-interleaver order.

        :param array: That array that is needed to set/restore interleaver order.
        """
        pass

    def actf(self, inputs):
        """
        Activation functions that can be called from the inherited class.

        :param inputs: Input tensors.
        :return: Output tensor to which the activation function is applied.
        """
        if self.args["coder_actf"] == "tanh":
            return torch.tanh(inputs)
        elif self.args["coder_actf"] == "elu":
            return func.elu(inputs)
        elif self.args["coder_actf"] == "relu":
            return torch.relu(inputs)
        elif self.args["coder_actf"] == "selu":
            return func.selu(inputs)
        elif self.args["coder_actf"] == "sigmoid":
            return torch.sigmoid(inputs)
        elif self.args["coder_actf"] == "identity":
            return inputs
        elif self.args["coder_actf"] == "leakyrelu":
            return func.leaky_relu(inputs)
        elif self.args["coder_actf"] == "gelu":
            return func.gelu(inputs)
        else:
            return inputs


# MLP Coder
class CoderMLP(CoderBase):
    def __init__(self, arguments):
        """
        MLP based network. Used to reconstruct the code stream from the encoder after
        applying the noisy channel.

        :param arguments: Arguments as dictionary.
        """
        super(CoderMLP, self).__init__(arguments)

        self._dropout = torch.nn.Dropout(self.args["coder_dropout"])

        self._linear_1 = torch.nn.Linear(self.args["block_length"] + self.args["block_padding"] + self.args["redundancy"],
                                         self.args["block_length"])
        self._linear_2 = torch.nn.Linear(self.args["block_length"] + self.args["block_padding"] + self.args["redundancy"],
                                         self.args["block_length"])
        if self.args["rate"] == "onethird":
            self._linear_3 = torch.nn.Linear(self.args["block_length"] + self.args["block_padding"] + self.args["redundancy"],
                                             self.args["block_length"])


    def set_parallel(self):
        """
        Ensures that forward and backward propagation operations can be performed on multiple GPUs.
        """
        self._linear_1 = torch.nn.DataParallel(self._linear_1)
        self._linear_2 = torch.nn.DataParallel(self._linear_2)
        if self.args["rate"] == "onethird":
            self._linear_3 = torch.nn.DataParallel(self._linear_3)

    def forward(self, inputs):
        """
        Calculates output tensors from input tensors based on the process.

        :param inputs: Input tensor.
        :return: Output tensor of coder.
        """
        x_sys = torch.flatten(inputs[:, :, 0], start_dim=1)
        x_sys = self.actf(self._dropout(self._linear_1(x_sys)))
        x_sys = x_sys.reshape((inputs.size()[0], self.args["block_length"], 1))

        x_p1 = torch.flatten(inputs[:, :, 1], start_dim=1)
        x_p1 = self.actf(self._dropout(self._linear_2(x_p1)))
        x_p1 = x_p1.reshape((inputs.size()[0], self.args["block_length"], 1))

        if self.args["rate"] == "onethird":
            x_p2 = torch.flatten(inputs[:, :, 2], start_dim=1)
            x_p2 = self.actf(self._dropout(self._linear_3(x_p2)))
            x_p2 = x_p2.reshape((inputs.size()[0], self.args["block_length"], 1))

            x = torch.cat([x_sys, x_p1, x_p2], dim=2)
        else:
            x = torch.cat([x_sys, x_p1], dim=2)

        if not self.args["channel"] == "continuous" or self.args["continuous_coder"]:
            x = Quantizer.apply(x)
        return x


# CNN Coder
class CoderCNN(CoderBase):
    def __init__(self, arguments):
        """
        CNN based network. Used to reconstruct the code stream from the encoder after
        applying the noisy channel.

        :param arguments: Arguments as dictionary.
        """
        super(CoderCNN, self).__init__(arguments)

        self._dropout = torch.nn.Dropout(self.args["coder_dropout"])

        self._cnn_1 = Conv1d(self.args["coder_actf"],
                             layers=self.args["coder_layers"],
                             in_channels=1,
                             out_channels=self.args["coder_units"],# + 16,
                             kernel_size=self.args["coder_kernel"])
        self._cnn_2 = Conv1d(self.args["coder_actf"],
                             layers=self.args["coder_layers"],
                             in_channels=1,
                             out_channels=self.args["coder_units"], #+16
                             kernel_size=self.args["coder_kernel"])

        self._linear_1 = torch.nn.Linear((self.args["coder_units"]) * (
                    self.args["block_length"] + int(self.args["redundancy"]) + self.args["block_padding"]),
                                         self.args["block_length"])
        self._linear_2 = torch.nn.Linear((self.args["coder_units"]) * (
                self.args["block_length"] + int(self.args["redundancy"]) + self.args["block_padding"]),
                                         self.args["block_length"])
        if self.args["batch_norm"]:
            self._batch_norm_1 = torch.nn.BatchNorm1d(self.args["block_length"])
            self._batch_norm_2 = torch.nn.BatchNorm1d(self.args["block_length"])
        if self.args["rate"] == "onethird":
            self._cnn_3 = Conv1d(self.args["coder_actf"],
                                 layers=self.args["coder_layers"],
                                 in_channels=1,
                                 out_channels=self.args["coder_units"], #+16
                                 kernel_size=self.args["coder_kernel"])
            self._linear_3 = torch.nn.Linear((self.args["coder_units"]) * (
                    self.args["block_length"] + int(self.args["redundancy"]) + self.args["block_padding"]),
                                             self.args["block_length"])
            if self.args["batch_norm"]:
                self._batch_norm_3 = torch.nn.BatchNorm1d(self.args["block_length"])

    def set_parallel(self):
        """
        Ensures that forward and backward propagation operations can be performed on multiple GPUs.
        """
        self._cnn_1 = torch.nn.DataParallel(self._cnn_1)
        self._cnn_2 = torch.nn.DataParallel(self._cnn_2)
        self._linear_1 = torch.nn.DataParallel(self._linear_1)
        self._linear_2 = torch.nn.DataParallel(self._linear_2)
        if self.args["rate"] == "onethird":
            self._cnn_3 = torch.nn.DataParallel(self._cnn_3)
            self._linear_3 = torch.nn.DataParallel(self._linear_3)

    def forward(self, inputs):
        """
        Calculates output tensors from input tensors based on the process.

        :param inputs: Input tensor.
        :return: Output tensor of coder.
        """
        x_sys = inputs[:, :, 0].view((inputs.size()[0], inputs.size()[1], 1))
        x_sys = self._cnn_1(x_sys)
        x_sys = torch.flatten(x_sys, start_dim=1)
        x_sys = self.actf(self._dropout(self._linear_1(x_sys)))

        x_sys = x_sys.reshape((inputs.size()[0], self.args["block_length"], 1))
        if self.args["batch_norm"]:
            x_sys = self._batch_norm_1(x_sys)

        x_p1 = inputs[:, :, 1].view((inputs.size()[0], inputs.size()[1], 1))
        x_p1 = self._cnn_2(x_p1)
        x_p1 = torch.flatten(x_p1, start_dim=1)
        x_p1 = self.actf(self._dropout(self._linear_2(x_p1)))
        x_p1 = x_p1.reshape((inputs.size()[0], self.args["block_length"], 1))
        if self.args["batch_norm"]:
            x_p1 = self._batch_norm_2(x_p1)

        if self.args["rate"] == "onethird":
            x_p2 = inputs[:, :, 2].view((inputs.size()[0], inputs.size()[1], 1))
            x_p2 = self._cnn_3(x_p2)
            x_p2 = torch.flatten(x_p2, start_dim=1)
            x_p2 = self.actf(self._dropout(self._linear_3(x_p2)))
            x_p2 = x_p2.reshape((inputs.size()[0], self.args["block_length"], 1))
            if self.args["batch_norm"]:
                x_p2 = self._batch_norm_3(x_p2)

            x = torch.cat([x_sys, x_p1, x_p2], dim=2)
        else:
            x = torch.cat([x_sys, x_p1], dim=2)
        if not self.args["channel"] == "continuous" or self.args["continuous_coder"]:
            x = Quantizer.apply(x)
        return x


# RNN Coder
class CoderRNN(CoderBase):
    def __init__(self, arguments):
        """
        RNN based network. Used to reconstruct the code stream from the encoder after
        applying the noisy channel.

        :param arguments: Arguments as dictionary.
        """
        super(CoderRNN, self).__init__(arguments)

        rnn = torch.nn.RNN
        if self.args["coder_rnn"].lower() == 'gru':
            rnn = torch.nn.GRU
        elif self.args["coder_rnn"].lower() == 'lstm':
            rnn = torch.nn.LSTM

        self._dropout = torch.nn.Dropout(self.args["coder_dropout"])

        self._rnn_1 = rnn(1, self.args["coder_units"],
                          num_layers=self.args["coder_layers"],
                          bias=True,
                          batch_first=True,
                          dropout=0,
                          bidirectional=True)
        self._linear_1 = torch.nn.Linear(2 * self.args["coder_units"] * (
                self.args["block_length"] + self.args["block_padding"] + self.args["redundancy"]),
                                         self.args["block_length"])
        self._rnn_2 = rnn(1, self.args["coder_units"],
                          num_layers=self.args["coder_layers"],
                          bias=True,
                          batch_first=True,
                          dropout=0,
                          bidirectional=True)
        self._linear_2 = torch.nn.Linear(2 * self.args["coder_units"] * (
                self.args["block_length"] + self.args["block_padding"] + self.args["redundancy"]),
                                         self.args["block_length"])

        if self.args["rate"] == "onethird":
            self._rnn_3 = rnn(1, self.args["coder_units"],
                              num_layers=self.args["coder_layers"],
                              bias=True,
                              batch_first=True,
                              dropout=0,
                              bidirectional=True)
            self._linear_3 = torch.nn.Linear(2 * self.args["coder_units"] * (
                    self.args["block_length"] + self.args["block_padding"] + self.args["redundancy"]),
                                             self.args["block_length"])

    def set_parallel(self):
        """
        Ensures that forward and backward propagation operations can be performed on multiple GPUs.
        """
        self._rnn_1 = torch.nn.DataParallel(self._rnn_1)
        self._rnn_2 = torch.nn.DataParallel(self._rnn_2)
        self._linear_1 = torch.nn.DataParallel(self._linear_1)
        self._linear_2 = torch.nn.DataParallel(self._linear_2)
        if self.args["rate"] == "onethird":
            self._rnn_3 = torch.nn.DataParallel(self._rnn_3)
            self._linear_3 = torch.nn.DataParallel(self._linear_3)

    def forward(self, inputs):
        """
        Calculates output tensors from input tensors based on the process.

        :param inputs: Input tensor.
        :return: Output tensor of coder.
        """
        x_sys = inputs[:, :, 0].view((inputs.size()[0], inputs.size()[1], 1))
        x_sys, _ = self._rnn_1(x_sys)
        x_sys = torch.flatten(x_sys, start_dim=1)
        x_sys = self.actf(self._dropout(self._linear_1(x_sys)))
        x_sys = x_sys.reshape((inputs.size()[0], self.args["block_length"], 1))

        x_p1 = inputs[:, :, 1].view((inputs.size()[0], inputs.size()[1], 1))
        x_p1, _ = self._rnn_2(x_p1)
        x_p1 = torch.flatten(x_p1, start_dim=1)
        x_p1 = self.actf(self._dropout(self._linear_2(x_p1)))
        x_p1 = x_p1.reshape((inputs.size()[0], self.args["block_length"], 1))

        if self.args["rate"] == "onethird":
            x_p2 = inputs[:, :, 2].view((inputs.size()[0], inputs.size()[1], 1))
            x_p2, _ = self._rnn_3(x_p2)
            x_p2 = torch.flatten(x_p2, start_dim=1)
            x_p2 = self.actf(self._dropout(self._linear_3(x_p2)))
            x_p2 = x_p2.reshape((inputs.size()[0], self.args["block_length"], 1))

            x = torch.cat([x_sys, x_p1, x_p2], dim=2)
        else:
            x = torch.cat([x_sys, x_p1], dim=2)
        if not self.args["channel"] == "continuous" or self.args["continuous_coder"]:
            x = Quantizer.apply(x)
        return x

class CoderCNN_nolat(CoderBase):
    def __init__(self, arguments):
        """
        CNN based network. Used to reconstruct the code stream from the encoder after
        applying the noisy channel.
        :param arguments: Arguments as dictionary.
        """
        super(CoderCNN_nolat, self).__init__(arguments)

        self._dropout = torch.nn.Dropout(self.args["coder_dropout"])

        self._cnn_1 = Conv1d(self.args["coder_actf"],
                             layers=self.args["coder_layers"],
                             in_channels=1,
                             out_channels=self.args["coder_units"],
                             kernel_size=self.args["coder_kernel"])
        self._linear_1 = torch.nn.Linear(self.args["coder_units"] * (self.args["block_length"] + self.args["block_padding"]), self.args["block_length"])
        #self._linear_1 = torch.nn.Linear(self.args["coder_units"] * (self.args["block_length"]), self.args["block_length"])
        self._cnn_2 = Conv1d(self.args["coder_actf"],
                             layers=self.args["coder_layers"],
                             in_channels=1,
                             out_channels=self.args["coder_units"],
                             kernel_size=self.args["coder_kernel"])
        self._linear_2 = torch.nn.Linear(self.args["coder_units"] * (self.args["block_length"] + self.args["block_padding"]), self.args["block_length"])
        #self._linear_2 = torch.nn.Linear(
        #    self.args["coder_units"] * (self.args["block_length"]),
        #    self.args["block_length"])
        if self.args["rate"] == "onethird":
            self._cnn_3 = Conv1d(self.args["coder_actf"],
                                 layers=self.args["coder_layers"],
                                 in_channels=1,
                                 out_channels=self.args["coder_units"],
                                 kernel_size=self.args["coder_kernel"])
            self._linear_3 = torch.nn.Linear(self.args["coder_units"] * (self.args["block_length"] + self.args["block_padding"]), self.args["block_length"])
            #self._linear_3 = torch.nn.Linear(
            #    self.args["coder_units"] * (self.args["block_length"]),
            #    self.args["block_length"])

    def set_parallel(self):
        """
        Ensures that forward and backward propagation operations can be performed on multiple GPUs.
        """
        self._cnn_1 = torch.nn.DataParallel(self._cnn_1)
        self._cnn_2 = torch.nn.DataParallel(self._cnn_2)
        self._linear_1 = torch.nn.DataParallel(self._linear_1)
        self._linear_2 = torch.nn.DataParallel(self._linear_2)
        if self.args["rate"] == "onethird":
            self._cnn_3 = torch.nn.DataParallel(self._cnn_3)
            self._linear_3 = torch.nn.DataParallel(self._linear_3)

    def forward(self, inputs):
        """
        Calculates output tensors from input tensors based on the process.
        :param inputs: Input tensor.
        :return: Output tensor of coder.
        """

        x_sys = inputs[:, :, 0].view((inputs.size()[0], inputs.size()[1], 1))
        x_sys = self._cnn_1(x_sys)
        x_sys = torch.flatten(x_sys, start_dim=1)
        x_sys = self.actf(self._dropout(self._linear_1(x_sys)))
        x_sys = x_sys.reshape((inputs.size()[0], self.args["block_length"], 1))

        x_p1 = inputs[:, :, 1].view((inputs.size()[0], inputs.size()[1], 1))
        x_p1 = self._cnn_2(x_p1)
        x_p1 = torch.flatten(x_p1, start_dim=1)
        x_p1 = self.actf(self._dropout(self._linear_2(x_p1)))
        x_p1 = x_p1.reshape((inputs.size()[0], self.args["block_length"], 1))

        if self.args["rate"] == "onethird":
            x_p2 = inputs[:, :, 2].view((inputs.size()[0], inputs.size()[1], 1))
            x_p2 = self._cnn_3(x_p2)
            x_p2 = torch.flatten(x_p2, start_dim=1)
            x_p2 = self.actf(self._dropout(self._linear_3(x_p2)))
            x_p2 = x_p2.reshape((inputs.size()[0], self.args["block_length"], 1))

            x = torch.cat([x_sys, x_p1, x_p2], dim=2)
        else:
            x = torch.cat([x_sys, x_p1], dim=2)
        if not self.args["channel"] == "continuous" or self.args["continuous_coder"]:
            x = Quantizer.apply(x)
        return x

class CoderCNN_RNN(CoderBase):
    def __init__(self, arguments):
        """
        CNN based network. Used to reconstruct the code stream from the encoder after
        applying the noisy channel.
        :param arguments: Arguments as dictionary.
        """
        super(CoderCNN_RNN, self).__init__(arguments)

        self._dropout = torch.nn.Dropout(self.args["coder_dropout"])

        self._cnn_1 = Conv1d(self.args["coder_actf"],
                             layers=self.args["coder_layers"],
                             in_channels=1,
                             out_channels=self.args["coder_units"],
                             kernel_size=self.args["coder_kernel"])
        self._rnn_1 = torch.nn.GRU(input_size=self.args["coder_units"],
                                   hidden_size=self.args["coder_units"],
                                   num_layers=self.args["coder_layers"],
                                   batch_first=True)
        self._linear_1 = torch.nn.Linear(self.args["coder_units"] * (
                self.args["block_length"] + self.args["block_padding"] + self.args["redundancy"]),
                                         self.args["block_length"])
        self._cnn_2 = Conv1d(self.args["coder_actf"],
                             layers=self.args["coder_layers"],
                             in_channels=1,
                             out_channels=self.args["coder_units"],
                             kernel_size=self.args["coder_kernel"])
        self._rnn_2 = torch.nn.GRU(input_size=self.args["coder_units"],
                                   hidden_size=self.args["coder_units"],
                                   num_layers=self.args["coder_layers"],
                                   batch_first=True)
        self._linear_2 = torch.nn.Linear(self.args["coder_units"] * (
                self.args["block_length"] + self.args["block_padding"] + self.args["redundancy"]),
                                         self.args["block_length"])
        if self.args["batch_norm"]:
            self._batch_norm_1 = torch.nn.BatchNorm1d(self.args["block_length"] + self.args["block_padding"]
                                                      + self.args["redundancy"])
            self._batch_norm_2 = torch.nn.BatchNorm1d(self.args["block_length"] + self.args["block_padding"]
                                                      + self.args["redundancy"])
        if self.args["rate"] == "onethird":
            self._cnn_3 = Conv1d(self.args["coder_actf"],
                                 layers=self.args["coder_layers"],
                                 in_channels=1,
                                 out_channels=self.args["coder_units"],
                                 kernel_size=self.args["coder_kernel"])
            self._rnn_3 = torch.nn.GRU(input_size=self.args["coder_units"],
                                       hidden_size=self.args["coder_units"],
                                       num_layers=self.args["coder_layers"],
                                       batch_first=True)
            self._linear_3 = torch.nn.Linear(self.args["coder_units"] * (
                    self.args["block_length"] + self.args["block_padding"] + self.args["redundancy"]),
                                             self.args["block_length"])
            if self.args["batch_norm"]:
                self._batch_norm_3 = torch.nn.BatchNorm1d(self.args["block_length"] + self.args["block_padding"]
                                                          + self.args["redundancy"])
    def set_parallel(self):
        """
        Ensures that forward and backward propagation operations can be performed on multiple GPUs.
        """
        self._cnn_1 = torch.nn.DataParallel(self._cnn_1)
        self._rnn_1 = torch.nn.DataParallel(self._rnn_1)
        self._cnn_2 = torch.nn.DataParallel(self._cnn_2)
        self._rnn_2 = torch.nn.DataParallel(self._rnn_2)
        self._linear_1 = torch.nn.DataParallel(self._linear_1)
        self._linear_2 = torch.nn.DataParallel(self._linear_2)
        if self.args["batch_norm"]:
            self._batch_norm_1 = torch.nn.DataParallel(self._batch_norm_1)
            self._batch_norm_2 = torch.nn.DataParallel(self._batch_norm_2)
        if self.args["rate"] == "onethird":
            self._cnn_3 = torch.nn.DataParallel(self._cnn_3)
            self._rnn_3 = torch.nn.DataParallel(self._rnn_3)
            self._linear_3 = torch.nn.DataParallel(self._linear_3)
            if self.args["batch_norm"]:
                self._batch_norm_3 = torch.nn.DataParallel(self._batch_norm_3)

    def forward(self, inputs):
        """
        Calculates output tensors from input tensors based on the process.
        :param inputs: Input tensor.
        :return: Output tensor of coder.
        """
        x_sys = inputs[:, :, 0].view((inputs.size()[0], inputs.size()[1], 1))
        x_sys = self._cnn_1(x_sys)
        x_sys, _ = self._rnn_1(x_sys)
        if self.args["batch_norm"]:
            x_sys = self._batch_norm_1(x_sys)
        x_sys = torch.flatten(x_sys, start_dim=1)
        x_sys = self.actf(self._dropout(self._linear_1(x_sys)))
        x_sys = x_sys.reshape((inputs.size()[0], self.args["block_length"], 1))

        x_p1 = inputs[:, :, 1].view((inputs.size()[0], inputs.size()[1], 1))
        x_p1 = self._cnn_2(x_p1)
        x_p1, _ = self._rnn_1(x_p1)
        if self.args["batch_norm"]:
            x_p1 = self._batch_norm_1(x_p1)
        x_p1 = torch.flatten(x_p1, start_dim=1)
        x_p1 = self.actf(self._dropout(self._linear_2(x_p1)))
        x_p1 = x_p1.reshape((inputs.size()[0], self.args["block_length"], 1))

        if self.args["rate"] == "onethird":
            x_p2 = inputs[:, :, 2].view((inputs.size()[0], inputs.size()[1], 1))
            x_p2 = self._cnn_3(x_p2)
            x_p2, _ = self._rnn_3(x_p2)
            if self.args["batch_norm"]:
                x_p2 = self._batch_norm_1(x_p2)
            x_p2 = torch.flatten(x_p2, start_dim=1)
            x_p2 = self.actf(self._dropout(self._linear_3(x_p2)))
            x_p2 = x_p2.reshape((inputs.size()[0], self.args["block_length"], 1))

            x = torch.cat([x_sys, x_p1, x_p2], dim=2)
        else:
            x = torch.cat([x_sys, x_p1], dim=2)
        if not self.args["channel"] == "continuous" or self.args["continuous_coder"]:
            x = Quantizer.apply(x)
        return x

class CoderTransformer(CoderBase):
    def __init__(self, arguments):
        """
        Transformer based network. Used to reconstruct the code stream from the encoder after
        applying the noisy channel.

        :param arguments: Arguments as dictionary.
        """
        super(CoderTransformer, self).__init__(arguments)

        self._dropout = torch.nn.Dropout(self.args["coder_dropout"])

        self._encoder_layer_1 = torch.nn.TransformerEncoderLayer(d_model=self.args["coder_units"],
                                                                 nhead=self.args["coder_kernel"],
                                                                 dropout=self.args["coder_dropout"],
                                                                 activation='relu',
                                                                 batch_first=True)
        self._transformer_1 = torch.nn.TransformerEncoder(self._encoder_layer_1, num_layers=self.args["coder_layers"])
        self._linear_1 = torch.nn.Linear(
            self.args["coder_units"] * (self.args["block_length"] + self.args["block_padding"]
                                        + self.args["redundancy"]), self.args["block_length"])

        self._encoder_layer_2 = torch.nn.TransformerEncoderLayer(d_model=self.args["coder_units"],
                                                                 nhead=self.args["coder_kernel"],
                                                                 dropout=self.args["coder_dropout"],
                                                                 activation='relu',
                                                                 batch_first=True)
        self._transformer_2 = torch.nn.TransformerEncoder(self._encoder_layer_2, num_layers=self.args["coder_layers"])
        self._linear_2 = torch.nn.Linear(
            self.args["coder_units"] * (self.args["block_length"] + self.args["block_padding"]
                                        + self.args["redundancy"]), self.args["block_length"])

        if self.args["rate"] == "onethird":
            self._encoder_layer_3 = torch.nn.TransformerEncoderLayer(d_model=self.args["coder_units"],
                                                                     nhead=self.args["coder_kernel"],
                                                                     dropout=self.args["coder_dropout"],
                                                                     activation='relu',
                                                                     batch_first=True)
            self._transformer_3 = torch.nn.TransformerEncoder(self._encoder_layer_3, num_layers=self.args["coder_layers"])
            self._linear_3 = torch.nn.Linear(
                self.args["coder_units"] * (self.args["block_length"] + self.args["block_padding"]
                                            + self.args["redundancy"]), self.args["block_length"])

    def set_parallel(self):
        """
        Ensures that forward and backward propagation operations can be performed on multiple GPUs.
        """
        self._transformer_1 = torch.nn.DataParallel(self._transformer_1)
        self._transformer_2 = torch.nn.DataParallel(self._transformer_2)
        self._linear_1 = torch.nn.DataParallel(self._linear_1)
        self._linear_2 = torch.nn.DataParallel(self._linear_2)
        if self.args["rate"] == "onethird":
            self._transformer_3 = torch.nn.DataParallel(self._transformer_3)
            self._linear_3 = torch.nn.DataParallel(self._linear_3)

    def forward(self, inputs):
        """
        Calculates output tensors from input tensors based on the process.

        :param inputs: Input tensor.
        :return: Output tensor of coder.
        """
        x_sys = inputs[:, :, 0].view((inputs.size()[0], inputs.size()[1], 1))
        x_sys = self._transformer_1(x_sys)
        x_sys = torch.flatten(x_sys, start_dim=1)
        x_sys = self.actf(self._dropout(self._linear_1(x_sys)))
        x_sys = x_sys.reshape((inputs.size()[0], self.args["block_length"], 1))

        x_p1 = inputs[:, :, 1].view((inputs.size()[0], inputs.size()[1], 1))
        x_p1 = self._transformer_2(x_p1)
        x_p1 = torch.flatten(x_p1, start_dim=1)
        x_p1 = self.actf(self._dropout(self._linear_2(x_p1)))
        x_p1 = x_p1.reshape((inputs.size()[0], self.args["block_length"], 1))

        if self.args["rate"] == "onethird":
            x_p2 = inputs[:, :, 2].view((inputs.size()[0], inputs.size()[1], 1))
            x_p2 = self._transformer_3(x_p2)
            x_p2 = torch.flatten(x_p2, start_dim=1)
            x_p2 = self.actf(self._dropout(self._linear_3(x_p2)))
            x_p2 = x_p2.reshape((inputs.size()[0], self.args["block_length"], 1))
            x = torch.cat((x_sys, x_p1, x_p2), dim=2)
        else:
            x = torch.cat((x_sys, x_p1), dim=2)
        if not self.args["channel"] == "continuous" or self.args["continuous_coder"]:
            x = Quantizer.apply(x)
        return x

class CoderCNN_conc(CoderBase):
    def __init__(self, arguments):
        """
        CNN based network. Used to reconstruct the code stream from the encoder after
        applying the noisy channel.
        :param arguments: Arguments as dictionary.
        """
        super(CoderCNN_conc, self).__init__(arguments)

        self._dropout = torch.nn.Dropout(self.args["coder_dropout"])
        self._cnn = Conv1d(self.args["coder_actf"],
                             layers=self.args["coder_layers"],
                             in_channels=1,
                             out_channels=self.args["coder_units"],
                             kernel_size=self.args["coder_kernel"])
        '''
        self._cnn = Conv1d_inc_kernel(self.args["coder_actf"],
                           layers=self.args["coder_layers"],
                           in_channels=1,
                           out_channels=self.args["coder_units"],
                           kernel_size=self.args["coder_kernel"])
        '''
        self._linear = torch.nn.Linear(self.args["coder_units"] * 3 * (
                self.args["block_length"] + self.args["block_padding"] + self.args["redundancy"]),
                                       self.args["block_length"]*3)

    def set_parallel(self):
        """
        Ensures that forward and backward propagation operations can be performed on multiple GPUs.
        """
        self._cnn = torch.nn.DataParallel(self._cnn)
        self._linear = torch.nn.DataParallel(self._linear)

    def forward(self, inputs):
        """
        Calculates output tensors from input tensors based on the process.
        :param inputs: Input tensor.
        :return: Output tensor of coder.
        """
        x = inputs.transpose(1, 2).reshape(self.args["batch_size"], -1, 1)
        x = self._cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.actf(self._dropout(self._linear(x)))
        x = x.reshape((inputs.size()[0], self.args["block_length"], 3))
        if not self.args["channel"] == "continuous" or self.args["continuous_coder"]:
            x = Quantizer.apply(x)
        return x


class EnsembleCNN(CoderBase):
    def __init__(self, arguments):
        """
        Ensemble of CNN based networks. Used to reconstruct the code stream from the encoder after
        applying the noisy channel.
        :param arguments: Arguments as dictionary.
        """
        super(EnsembleCNN, self).__init__(arguments)

        self.n_models = self.args['n_models']
        self.models = torch.nn.ModuleList([CoderCNN_nolat(self.args) for i in range(self.n_models)])

    def forward(self, inputs):
        """
        Calculates output tensors from input tensors based on the process.
        :param inputs: Input tensor.
        :return: Output tensor of coder.
        """
        '''
        outputs = []
        for i in range(self.n_models):
            outputs.append(self.models[i].forward(inputs).detach().numpy())
        #x = torch.mean(torch.stack(outputs), dim=0)
        x = np.apply_along_axis(lambda y: np.argmax(np.bincount(y)), axis=0, arr=outputs)
        #x = self.ensemble_predict()
        '''
        outputs = []
        for i in range(self.n_models):
            outputs.append(self.models[i].forward(inputs))

        outputs = torch.stack(outputs)  # shape: (n_models, batch_size, sequence_length, num_classes)
        x = torch.mode(outputs, dim=0).values  # shape: (batch_size, sequence_length, num_classes)
        return x

    def ensemble_predict(self, inp, models):
        outputs = []
        for model in models:
            model.eval()
            with torch.no_grad():
                output = model(inp)
                output = torch.round(output).permute(2, 0, 1)
                outputs.append(output)
        outputs = torch.stack(outputs, dim=0)
        majority = torch.mode(outputs, dim=0).values.permute(1, 2, 0)
        return majority


class CoderIDT(CoderBase):
    def __init__(self, arguments):
        """
        CNN based network. Used to reconstruct the code stream from the encoder after
        applying the noisy channel.
        :param arguments: Arguments as dictionary.
        """
        super(CoderIDT, self).__init__(arguments)

        self._dropout = torch.nn.Dropout(self.args["coder_dropout"])

        self._idl_1 = IDTLayer(self.args["block_length"]+self.args["block_padding"]+ self.args["redundancy"],
                               self.args["block_length"]+self.args["block_padding"]+ self.args["redundancy"], 2, 3, 1)
        #self._idl_1 = IDTLayer(self.args["block_length"], self.args["block_length"])
        self._cnn_1 = Conv1d(self.args["coder_actf"],
                             layers=self.args["coder_layers"],
                             in_channels=1,
                             out_channels=self.args["coder_units"],
                             kernel_size=self.args["coder_kernel"])
        self._linear_1 = torch.nn.Linear(self.args["coder_units"] * (
                self.args["block_length"] + self.args["block_padding"] + self.args["redundancy"]),
                                         self.args["block_length"])
        #self._linear_1 = torch.nn.Linear(self.args["coder_units"] * (self.args["block_length"]), self.args["block_length"])
        self._idl_2 = IDTLayer(self.args["block_length"]+self.args["block_padding"]+ self.args["redundancy"],
                               self.args["block_length"]+self.args["block_padding"]+ self.args["redundancy"], 2, 3, 1)
        #self._idl_2 = IDTLayer(self.args["block_length"], self.args["block_length"])
        self._cnn_2 = Conv1d(self.args["coder_actf"],
                             layers=self.args["coder_layers"],
                             in_channels=1,
                             out_channels=self.args["coder_units"],
                             kernel_size=self.args["coder_kernel"])
        self._linear_2 = torch.nn.Linear(self.args["coder_units"] * (
                self.args["block_length"] + self.args["block_padding"] + self.args["redundancy"]),
                                         self.args["block_length"])
        #self._linear_2 = torch.nn.Linear(
        #    self.args["coder_units"] * (self.args["block_length"]),
        #    self.args["block_length"])
        if self.args["rate"] == "onethird":
            self._idl_3 = IDTLayer(self.args["block_length"]+self.args["block_padding"] + self.args["redundancy"],
                                   self.args["block_length"]+self.args["block_padding"] + self.args["redundancy"], 2, 3, 1)
            #self._idl_3 = IDTLayer(self.args["block_length"], self.args["block_length"])
            self._cnn_3 = Conv1d(self.args["coder_actf"],
                                 layers=self.args["coder_layers"],
                                 in_channels=1,
                                 out_channels=self.args["coder_units"],
                                 kernel_size=self.args["coder_kernel"])
            self._linear_3 = torch.nn.Linear(self.args["coder_units"] * (
                    self.args["block_length"] + self.args["block_padding"] + self.args["redundancy"]),
                                             self.args["block_length"])
            #self._linear_3 = torch.nn.Linear(
            #    self.args["coder_units"] * (self.args["block_length"]),
            #    self.args["block_length"])


    def set_parallel(self):
        """
        Ensures that forward and backward propagation operations can be performed on multiple GPUs.
        """
        self._cnn_1 = torch.nn.DataParallel(self._cnn_1)
        self._cnn_2 = torch.nn.DataParallel(self._cnn_2)
        self._linear_1 = torch.nn.DataParallel(self._linear_1)
        self._linear_2 = torch.nn.DataParallel(self._linear_2)
        if self.args["rate"] == "onethird":
            self._cnn_3 = torch.nn.DataParallel(self._cnn_3)
            self._linear_3 = torch.nn.DataParallel(self._linear_3)

    def forward(self, inputs, target):
        """
        Calculates output tensors from input tensors based on the process.
        :param inputs: Input tensor.
        :return: Output tensor of coder.
        """
        x_sys = inputs[:, :, 0].view((inputs.size()[0], inputs.size()[1], 1))
        target_x_sys = target[:, :, 0].view((target.size()[0], target.size()[1], 1))
        x_sys =self._idl_1(x_sys, target_x_sys)
        x_sys = self._cnn_1(x_sys)
        x_sys = torch.flatten(x_sys, start_dim=1)
        x_sys = self.actf(self._dropout(self._linear_1(x_sys)))
        x_sys = x_sys.reshape((inputs.size()[0], self.args["block_length"], 1))

        x_p1 = inputs[:, :, 1].view((inputs.size()[0], inputs.size()[1], 1))
        target_x_p1 = target[:, :, 1].view((target.size()[0], target.size()[1], 1))
        x_p1 =self._idl_2(x_p1, target_x_p1)
        x_p1 = self._cnn_2(x_p1)
        x_p1 = torch.flatten(x_p1, start_dim=1)
        x_p1 = self.actf(self._dropout(self._linear_2(x_p1)))
        x_p1 = x_p1.reshape((inputs.size()[0], self.args["block_length"], 1))

        if self.args["rate"] == "onethird":
            x_p2 = inputs[:, :, 2].view((inputs.size()[0], inputs.size()[1], 1))
            target_x_p2 = target[:, :, 2].view((target.size()[0], target.size()[1], 1))
            x_p2 = self._idl_2(x_p2, target_x_p2)
            x_p2 = self._cnn_3(x_p2)
            x_p2 = torch.flatten(x_p2, start_dim=1)
            x_p2 = self.actf(self._dropout(self._linear_3(x_p2)))
            x_p2 = x_p2.reshape((inputs.size()[0], self.args["block_length"], 1))

            x = torch.cat([x_sys, x_p1, x_p2], dim=2)
        else:
            x = torch.cat([x_sys, x_p1], dim=2)
        if not self.args["channel"] == "continuous" or self.args["continuous_coder"]:
            x = Quantizer.apply(x)
        return x

class ResNetBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ResNetBlock, self).__init__()

        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = torch.nn.BatchNorm1d(out_channels)
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = torch.nn.BatchNorm1d(out_channels)

        if in_channels == out_channels:
            self.shortcut = torch.nn.Identity()
        else:
            self.shortcut = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = func.relu(x, inplace=True)

        x = self.conv2(x)
        x = self.bn2(x)

        x += shortcut
        x = func.relu(x, inplace=True)

        return x

class ResNetCoder(CoderBase):
    def __init__(self, arguments):
        super(ResNetCoder, self).__init__(arguments)

        self._dropout = torch.nn.Dropout(self.args["coder_dropout"])

        self._linear_1 = torch.nn.Linear((self.args["block_length"] + self.args["block_padding"]
                                          + self.args["redundancy"]), self.args["block_length"]) #+16
        self._linear_2 = torch.nn.Linear((self.args["block_length"] + self.args["block_padding"]
                                          + self.args["redundancy"]), self.args["block_length"]) #+16
        if self.args["rate"] == "onethird":
            self._linear_3 = torch.nn.Linear((self.args["block_length"] + self.args["block_padding"]
                                              + self.args["redundancy"]),self.args["block_length"])  # +16

        self._cnn_1 = torch.nn.Conv1d((self.args["block_length"] + self.args["block_padding"]
                                       + self.args["redundancy"]), self.args["coder_units"], kernel_size=3, padding=1)
        self._bn_1 = torch.nn.BatchNorm1d(self.args["coder_units"])

        layers = []
        for i in range(self.args["coder_layers"]):
            layers.append(ResNetBlock(self.args["coder_units"], self.args["coder_units"]))
        self._layers = torch.nn.Sequential(*layers)

        self._cnn_2 = torch.nn.Conv1d(self.args["coder_units"],
                                      (self.args["block_length"] + self.args["block_padding"]
                                       + self.args["redundancy"]), kernel_size=7, padding=4)

    def set_parallel(self):
        """
        Ensures that forward and backward propagation operations can be performed on multiple GPUs.
        """
        self._cnn_1 = torch.nn.DataParallel(self._cnn_1)
        self._cnn_2 = torch.nn.DataParallel(self._cnn_2)
        self._linear_1 = torch.nn.DataParallel(self._linear_1)
        self._linear_2 = torch.nn.DataParallel(self._linear_2)
        if self.args["rate"] == "onethird":
            self._linear_3 = torch.nn.DataParallel(self._linear_3)

    def forward(self, inputs):
        x = self._cnn_1(inputs)
        x = self._bn_1(x)
        x = func.relu(x, inplace=True)

        x = self._layers(x)

        x = self._cnn_2(x)

        x_sys = x[:, :, 0].view((x.size()[0], x.size()[1], 1))
        x_p1 = x[:, :, 1].view((x.size()[0], x.size()[1], 1))

        x_sys = torch.flatten(x_sys, start_dim=1)
        x_sys = self.actf(self._dropout(self._linear_1(x_sys)))
        x_sys = x_sys.reshape((inputs.size()[0], self.args["block_length"], 1))

        x_p1 = torch.flatten(x_p1, start_dim=1)
        x_p1 = self.actf(self._dropout(self._linear_2(x_p1)))
        x_p1 = x_p1.reshape((inputs.size()[0], self.args["block_length"], 1))

        if self.args["rate"] == "onethird":
            x_p2 = inputs[:, :, 2].view((x.size()[0], x.size()[1], 1))

            x_p2 = torch.flatten(x_p2, start_dim=1)
            x_p2 = self.actf(self._dropout(self._linear_3(x_p2)))
            x_p2 = x_p2.reshape((inputs.size()[0], self.args["block_length"], 1))

            x = torch.cat([x_sys, x_p1, x_p2], dim=2)
        else:
            x = torch.cat([x_sys, x_p1], dim=2)
        if not self.args["channel"] == "continuous" or self.args["continuous_coder"]:
            x = Quantizer.apply(x)

        return x
'''

class ResNetCoder(CoderBase):
    def __init__(self, arguments):
        super(ResNetCoder, self).__init__(arguments)

        self._dropout = torch.nn.Dropout(self.args["coder_dropout"])

        self._linear_1 = torch.nn.Linear((self.args["block_length"]), self.args["block_length"]) #+16
        self._linear_2 = torch.nn.Linear((self.args["block_length"]), self.args["block_length"]) #+16
        if self.args["rate"] == "onethird":
            self._linear_3 = torch.nn.Linear((self.args["block_length"]),self.args["block_length"])  # +16

        self._cnn_1 = torch.nn.Conv1d((self.args["block_length"]), self.args["coder_units"], kernel_size=3, padding=1)
        self._bn_1 = torch.nn.BatchNorm1d(self.args["coder_units"])

        layers = []
        for i in range(self.args["coder_layers"]):
            layers.append(ResNetBlock(self.args["coder_units"], self.args["coder_units"]))
        self._layers = torch.nn.Sequential(*layers)

        self._cnn_2 = torch.nn.Conv1d(self.args["coder_units"], (self.args["block_length"]), kernel_size=7, padding=4)

    def set_parallel(self):
        """
        Ensures that forward and backward propagation operations can be performed on multiple GPUs.
        """
        self._cnn_1 = torch.nn.DataParallel(self._cnn_1)
        self._cnn_2 = torch.nn.DataParallel(self._cnn_2)
        self._linear_1 = torch.nn.DataParallel(self._linear_1)
        self._linear_2 = torch.nn.DataParallel(self._linear_2)
        if self.args["rate"] == "onethird":
            self._linear_3 = torch.nn.DataParallel(self._linear_3)

    def forward(self, inputs):
        x = self._cnn_1(inputs)
        x = self._bn_1(x)
        x = func.relu(x, inplace=True)

        x = self._layers(x)

        x = self._cnn_2(x)

        x_sys = x[:, :, 0].view((x.size()[0], x.size()[1], 1))
        x_p1 = x[:, :, 1].view((x.size()[0], x.size()[1], 1))

        x_sys = torch.flatten(x_sys, start_dim=1)
        x_sys = self.actf(self._dropout(self._linear_1(x_sys)))
        x_sys = x_sys.reshape((inputs.size()[0], self.args["block_length"], 1))

        x_p1 = torch.flatten(x_p1, start_dim=1)
        x_p1 = self.actf(self._dropout(self._linear_2(x_p1)))
        x_p1 = x_p1.reshape((inputs.size()[0], self.args["block_length"], 1))

        if self.args["rate"] == "onethird":
            x_p2 = inputs[:, :, 2].view((x.size()[0], x.size()[1], 1))

            x_p2 = torch.flatten(x_p2, start_dim=1)
            x_p2 = self.actf(self._dropout(self._linear_3(x_p2)))
            x_p2 = x_p2.reshape((inputs.size()[0], self.args["block_length"], 1))

            x = torch.cat([x_sys, x_p1, x_p2], dim=2)
        else:
            x = torch.cat([x_sys, x_p1], dim=2)
        if not self.args["channel"] == "continuous" or self.args["continuous_coder"]:
            x = Quantizer.apply(x)

        return x
'''
class ResNetCoder_lat(CoderBase):
    def __init__(self, arguments):
        super(ResNetCoder_lat, self).__init__(arguments)

        self._dropout = torch.nn.Dropout(self.args["coder_dropout"])

        self._linear_1 = torch.nn.Linear((self.args["block_length"] + self.args["block_padding"]+ self.args["redundancy"]), self.args["block_length"]) #+16
        self._linear_2 = torch.nn.Linear((self.args["block_length"] + self.args["block_padding"]+ self.args["redundancy"]), self.args["block_length"]) #+16
        if self.args["rate"] == "onethird":
            self._linear_3 = torch.nn.Linear((self.args["block_length"] + self.args["block_padding"]+ self.args["redundancy"]),self.args["block_length"])  # +16

        self._cnn_1 = torch.nn.Conv1d((self.args["block_length"] + self.args["block_padding"] + self.args["redundancy"]), self.args["coder_units"], kernel_size=3, padding=1)
        self._bn_1 = torch.nn.BatchNorm1d(self.args["coder_units"])

        layers = []
        for i in range(self.args["coder_layers"]):
            layers.append(ResNetBlock(self.args["coder_units"], self.args["coder_units"]))
        self._layers = torch.nn.Sequential(*layers)

        self._cnn_2 = torch.nn.Conv1d(self.args["coder_units"], (self.args["block_length"] + self.args["block_padding"] + self.args["redundancy"]), kernel_size=7, padding=4)

    def set_parallel(self):
        """
        Ensures that forward and backward propagation operations can be performed on multiple GPUs.
        """
        self._cnn_1 = torch.nn.DataParallel(self._cnn_1)
        self._cnn_2 = torch.nn.DataParallel(self._cnn_2)
        self._linear_1 = torch.nn.DataParallel(self._linear_1)
        self._linear_2 = torch.nn.DataParallel(self._linear_2)
        if self.args["rate"] == "onethird":
            self._linear_3 = torch.nn.DataParallel(self._linear_3)

    def forward(self, inputs):
        x = self._cnn_1(inputs)
        x = self._bn_1(x)
        x = func.relu(x, inplace=True)

        x = self._layers(x)

        x = self._cnn_2(x)

        x_sys = x[:, :, 0].view((x.size()[0], x.size()[1], 1))
        x_p1 = x[:, :, 1].view((x.size()[0], x.size()[1], 1))

        x_sys = torch.flatten(x_sys, start_dim=1)
        x_sys = self.actf(self._dropout(self._linear_1(x_sys)))
        x_sys = x_sys.reshape((inputs.size()[0], self.args["block_length"], 1))

        x_p1 = torch.flatten(x_p1, start_dim=1)
        x_p1 = self.actf(self._dropout(self._linear_2(x_p1)))
        x_p1 = x_p1.reshape((inputs.size()[0], self.args["block_length"], 1))

        if self.args["rate"] == "onethird":
            x_p2 = inputs[:, :, 2].view((x.size()[0], x.size()[1], 1))

            x_p2 = torch.flatten(x_p2, start_dim=1)
            x_p2 = self.actf(self._dropout(self._linear_3(x_p2)))
            x_p2 = x_p2.reshape((inputs.size()[0], self.args["block_length"], 1))

            x = torch.cat([x_sys, x_p1, x_p2], dim=2)
        else:
            x = torch.cat([x_sys, x_p1], dim=2)
        if not self.args["channel"] == "continuous" or self.args["continuous_coder"]:
            x = Quantizer.apply(x)

        return x

class ResNetCoder_lat2(CoderBase):
    def __init__(self, arguments):
        super(ResNetCoder_lat2, self).__init__(arguments)

        self._dropout = torch.nn.Dropout(self.args["coder_dropout"])

        self._linear_1_1 = torch.nn.Linear((self.args["block_length"] + self.args["block_padding"]+ self.args["redundancy"]), self.args["block_length"]+(self.args["redundancy"]//2)) #+16
        self._linear_1_2 = torch.nn.Linear(
            (self.args["block_length"] + self.args["block_padding"] + (self.args["redundancy"] // 2)),
            self.args["block_length"])
        self._linear_2_1 = torch.nn.Linear((self.args["block_length"] + self.args["block_padding"]+ self.args["redundancy"]), self.args["block_length"]+(self.args["redundancy"]//2)) #+16
        self._linear_2_2 = torch.nn.Linear((self.args["block_length"] + self.args["block_padding"]
                                            + (self.args["redundancy"]//2)), self.args["block_length"]) #+16
        if self.args["rate"] == "onethird":
            self._linear_3_1 = torch.nn.Linear((self.args["block_length"] + self.args["block_padding"]+ self.args["redundancy"]),self.args["block_length"]+(self.args["redundancy"]//2))  # +16
            self._linear_3_2 = torch.nn.Linear((self.args["block_length"] + self.args["block_padding"]
                                                + (self.args["redundancy"]//2)),self.args["block_length"])  # +16
        self._cnn_1 = torch.nn.Conv1d((self.args["block_length"] + self.args["block_padding"] + self.args["redundancy"]), self.args["coder_units"], kernel_size=3, padding=1)
        self._bn_1 = torch.nn.BatchNorm1d(self.args["coder_units"])

        layers = []
        for i in range(self.args["coder_layers"]):
            layers.append(ResNetBlock(self.args["coder_units"], self.args["coder_units"]))
        self._layers = torch.nn.Sequential(*layers)

        self._cnn_2 = torch.nn.Conv1d(self.args["coder_units"], (self.args["block_length"] + self.args["block_padding"] + self.args["redundancy"]), kernel_size=7, padding=4)

    def set_parallel(self):
        """
        Ensures that forward and backward propagation operations can be performed on multiple GPUs.
        """
        self._cnn_1 = torch.nn.DataParallel(self._cnn_1)
        self._cnn_2 = torch.nn.DataParallel(self._cnn_2)
        self._linear_1 = torch.nn.DataParallel(self._linear_1)
        self._linear_2 = torch.nn.DataParallel(self._linear_2)
        if self.args["rate"] == "onethird":
            self._linear_3 = torch.nn.DataParallel(self._linear_3)

    def forward(self, inputs):
        x = self._cnn_1(inputs)
        x = self._bn_1(x)
        x = func.relu(x, inplace=True)

        x = self._layers(x)

        x = self._cnn_2(x)

        x_sys = x[:, :, 0].view((x.size()[0], x.size()[1], 1))
        x_p1 = x[:, :, 1].view((x.size()[0], x.size()[1], 1))

        x_sys = torch.flatten(x_sys, start_dim=1)
        x_sys = self.actf(self._dropout(self._linear_1_1(x_sys)))
        x_sys = self.actf(self._dropout(self._linear_1_2(x_sys)))
        x_sys = x_sys.reshape((inputs.size()[0], self.args["block_length"], 1))

        x_p1 = torch.flatten(x_p1, start_dim=1)
        x_p1 = self.actf(self._dropout(self._linear_2_1(x_p1)))
        x_p1 = self.actf(self._dropout(self._linear_2_2(x_p1)))
        x_p1 = x_p1.reshape((inputs.size()[0], self.args["block_length"], 1))

        if self.args["rate"] == "onethird":
            x_p2 = inputs[:, :, 2].view((x.size()[0], x.size()[1], 1))

            x_p2 = torch.flatten(x_p2, start_dim=1)
            x_p2 = self.actf(self._dropout(self._linear_3_1(x_p2)))
            x_p2 = self.actf(self._dropout(self._linear_3_2(x_p2)))
            x_p2 = x_p2.reshape((inputs.size()[0], self.args["block_length"], 1))

            x = torch.cat([x_sys, x_p1, x_p2], dim=2)
        else:
            x = torch.cat([x_sys, x_p1], dim=2)
        if not self.args["channel"] == "continuous" or self.args["continuous_coder"]:
            x = Quantizer.apply(x)

        return x

'''
class ResNetCoder_lat(CoderBase):
    def __init__(self, arguments):
        super(ResNetCoder_lat, self).__init__(arguments)

        self._dropout = torch.nn.Dropout(self.args["coder_dropout"])

        self._linear_1 = torch.nn.Linear((self.args["block_length"] + self.args["block_padding"]+ self.args["redundancy"]), self.args["block_length"]+int(self.args["redundancy"])) #+16
        self._linear_2 = torch.nn.Linear((self.args["block_length"] + self.args["block_padding"]+ self.args["redundancy"]), self.args["block_length"]+int(self.args["redundancy"])) #+16
        if self.args["rate"] == "onethird":
            self._linear_3 = torch.nn.Linear((self.args["block_length"] + self.args["block_padding"]+ self.args["redundancy"]),self.args["block_length"]+int(self.args["redundancy"]))  # +16

        self._cnn_1 = torch.nn.Conv1d((self.args["block_length"] + self.args["block_padding"] + self.args["redundancy"]), self.args["coder_units"], kernel_size=3, padding=1)
        self._bn_1 = torch.nn.BatchNorm1d(self.args["coder_units"])

        layers = []
        for i in range(self.args["coder_layers"]):
            layers.append(ResNetBlock(self.args["coder_units"], self.args["coder_units"]))
        self._layers = torch.nn.Sequential(*layers)

        self._cnn_2 = torch.nn.Conv1d(self.args["coder_units"], (self.args["block_length"] + self.args["block_padding"] + self.args["redundancy"]), kernel_size=7, padding=4)

    def set_parallel(self):
        """
        Ensures that forward and backward propagation operations can be performed on multiple GPUs.
        """
        self._cnn_1 = torch.nn.DataParallel(self._cnn_1)
        self._cnn_2 = torch.nn.DataParallel(self._cnn_2)
        self._linear_1 = torch.nn.DataParallel(self._linear_1)
        self._linear_2 = torch.nn.DataParallel(self._linear_2)
        if self.args["rate"] == "onethird":
            self._linear_3 = torch.nn.DataParallel(self._linear_3)

    def forward(self, inputs):
        x = self._cnn_1(inputs)
        x = self._bn_1(x)
        x = func.relu(x, inplace=True)

        x = self._layers(x)

        x = self._cnn_2(x)

        x_sys = x[:, :, 0].view((x.size()[0], x.size()[1], 1))
        x_p1 = x[:, :, 1].view((x.size()[0], x.size()[1], 1))

        x_sys = torch.flatten(x_sys, start_dim=1)
        x_sys = self.actf(self._dropout(self._linear_1(x_sys)))
        x_sys = x_sys.reshape((inputs.size()[0], self.args["block_length"]+self.args["redundancy"], 1))

        x_p1 = torch.flatten(x_p1, start_dim=1)
        x_p1 = self.actf(self._dropout(self._linear_2(x_p1)))
        x_p1 = x_p1.reshape((inputs.size()[0], self.args["block_length"]+self.args["redundancy"], 1))

        if self.args["rate"] == "onethird":
            x_p2 = inputs[:, :, 2].view((x.size()[0], x.size()[1], 1))

            x_p2 = torch.flatten(x_p2, start_dim=1)
            x_p2 = self.actf(self._dropout(self._linear_3(x_p2)))
            x_p2 = x_p2.reshape((inputs.size()[0], self.args["block_length"]+self.args["redundancy"], 1))

            x = torch.cat([x_sys, x_p1, x_p2], dim=2)
        else:
            x = torch.cat([x_sys, x_p1], dim=2)
        if not self.args["channel"] == "continuous" or self.args["continuous_coder"]:
            x = Quantizer.apply(x)

        return x

'''

class ResNetBlock2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetBlock2d, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=(1,1))
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=(1,1))
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

        if in_channels == out_channels:
            self.shortcut = torch.nn.Identity()
        else:
            self.shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=(1,1))


    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = func.relu(x, inplace=True)

        x = self.conv2(x)
        x = self.bn2(x)

        x += shortcut
        x = func.relu(x, inplace=True)

        return x

class ResNetCoder2d(CoderBase):
    def __init__(self, arguments):
        super(ResNetCoder2d, self).__init__(arguments)

        self._dropout = torch.nn.Dropout(self.args["coder_dropout"])

        self.interleaver = Interleaver()
        self.deinterleaver = DeInterleaver()

        self._linear_1 = torch.nn.Linear((self.args["block_length"] + self.args["block_padding"]
                                          + self.args["redundancy"]), self.args["block_length"])
        self._linear_2 = torch.nn.Linear((self.args["block_length"] + self.args["block_padding"]
                                          + self.args["redundancy"]), self.args["block_length"])
        self._linear_3 = torch.nn.Linear((self.args["block_length"] + self.args["block_padding"]
                                          + self.args["redundancy"]),self.args["block_length"])

        self._cnn_1 = torch.nn.Conv2d(in_channels=1, out_channels=self.args["coder_units"], kernel_size=3, stride=1, padding=1)
        self._bn_1 = torch.nn.BatchNorm2d(self.args["coder_units"])

        layers = []
        for i in range(self.args["coder_layers"]):
            layers.append(ResNetBlock2d(self.args["coder_units"], self.args["coder_units"]))
        self._layers = torch.nn.Sequential(*layers)

        self._cnn_2 = torch.nn.Conv2d(in_channels=self.args["coder_units"], out_channels=1, kernel_size=3, stride=1, padding=1)

    def set_interleaver_order(self, array):
        """
        Inheritance function to set the models interleaver order.
        :param array: That array that is needed to set interleaver order.
        """
        self.interleaver.set_order(array)
        self.deinterleaver.set_order(array)

    def set_parallel(self):
        """
        Ensures that forward and backward propagation operations can be performed on multiple GPUs.
        """
        self._cnn_1 = torch.nn.DataParallel(self._cnn_1)
        self._cnn_2 = torch.nn.DataParallel(self._cnn_2)
        self._linear_1 = torch.nn.DataParallel(self._linear_1)
        self._linear_2 = torch.nn.DataParallel(self._linear_2)
        if self.args["rate"] == "onethird":
            self._linear_3 = torch.nn.DataParallel(self._linear_3)

    def forward(self, inputs):

        x_sys = inputs[:, :, 0].view((inputs.size()[0], inputs.size()[1], 1))
        x_p1 = inputs[:, :, 1].view((inputs.size()[0], inputs.size()[1], 1))
        if self.args["rate"] == "onethird":
            x_p2 = inputs[:, :, 2].view((inputs.size()[0], inputs.size()[1], 1))

            x_p2_no_pad, x_p2_padding = torch.split(x_p2, self.args["block_length"], dim=1)
            x_p2_no_pad = self.deinterleaver(x_p2_no_pad)
            x_p2_deinter = torch.cat((x_p2_no_pad, x_p2_padding), dim=1)

            x_inp = torch.cat([x_sys, x_p1, x_p2_deinter], dim=2)
        else:
            x_inp = torch.cat([x_sys, x_p1], dim=2)

        x = x_inp.view(x_inp.size()[0], 1, x_inp.size()[1], x_inp.size()[2])
        x = self._cnn_1(x)
        x = self._bn_1(x)
        x = func.relu(x, inplace=True)

        x = self.layers(x)

        x = self._cnn_2(x)
        x = x.squeeze(1)
        x_sys = x[:, :, 0].view((x.size()[0], x.size()[1], 1))
        x_p1 = x[:, :, 1].view((x.size()[0], x.size()[1], 1))
        if self.args["rate"] == "onethird":
            x_p2 = x[:, :, 2].view((x.size()[0], x.size()[1], 1))

            x_p2_no_pad, x_p2_padding = torch.split(x_p2, self.args["block_length"], dim=1)
            x_p2_no_pad = self.interleaver(x_p2_no_pad)
            x_p2 = torch.cat((x_p2_no_pad, x_p2_padding), dim=1)

        x_sys = torch.flatten(x_sys, start_dim=1)
        x_sys = self.actf(self._dropout(self._linear_1(x_sys)))
        x_sys = x_sys.reshape((inputs.size()[0], self.args["block_length"], 1))

        x_p1 = torch.flatten(x_p1, start_dim=1)
        x_p1 = self.actf(self._dropout(self._linear_2(x_p1)))
        x_p1 = x_p1.reshape((inputs.size()[0], self.args["block_length"], 1))

        if self.args["rate"] == "onethird":
            x_p2 = torch.flatten(x_p2, start_dim=1)
            x_p2 = self.actf(self._dropout(self._linear_3(x_p2)))
            x_p2 = x_p2.reshape((inputs.size()[0], self.args["block_length"], 1))

            x = torch.cat([x_sys, x_p1, x_p2], dim=2)
        else:
            x = torch.cat([x_sys, x_p1])
        if not self.args["channel"] == "continuous" or self.args["continuous_coder"]:
            x = Quantizer.apply(x)

        return x

class ResNetCoder2d_1d(CoderBase):
    def __init__(self, arguments):
        super(ResNetCoder2d_1d, self).__init__(arguments)

        self._dropout = torch.nn.Dropout(self.args["coder_dropout"])

        self._linear_1 = torch.nn.Linear((self.args["block_length"] + self.args["block_padding"]
                                          + self.args["redundancy"]), self.args["block_length"])
        self._linear_2 = torch.nn.Linear((self.args["block_length"] + self.args["block_padding"]
                                          + self.args["redundancy"]), self.args["block_length"])
        self._linear_3 = torch.nn.Linear((self.args["block_length"] + self.args["block_padding"]
                                          + self.args["redundancy"]),self.args["block_length"])

        self._cnn_1_2d = torch.nn.Conv2d(in_channels=1, out_channels=self.args["coder_units"], kernel_size=3, stride=1, padding=1)
        self._bn_1_2d = torch.nn.BatchNorm2d(self.args["coder_units"])

        layers2d = []
        for i in range(self.args["coder_layers"]):
            layers2d.append(ResNetBlock2d(self.args["coder_units"], self.args["coder_units"]))
        self._layers_2d = torch.nn.Sequential(*layers2d)

        self._cnn_2_2d = torch.nn.Conv2d(in_channels=self.args["coder_units"], out_channels=1, kernel_size=3, stride=1, padding=1)

        self._cnn_1_1d = torch.nn.Conv1d(in_channels=1, out_channels=self.args["coder_units"], kernel_size=2, padding=1, stride=1)
        self._bn_1_1d = torch.nn.BatchNorm1d(self.args["coder_units"])

        layers_1d = []
        for i in range(self.args["coder_layers"]):
            layers_1d.append(ResNetBlock(self.args["coder_units"], self.args["coder_units"]))
        self._layers_1d = torch.nn.Sequential(*layers_1d)

        self._cnn_2_1d = torch.nn.Conv1d(self.args["coder_units"], 1, kernel_size=6, padding=2)
    def set_interleaver_order(self, array):
        """
        Inheritance function to set the models interleaver order.
        :param array: That array that is needed to set interleaver order.
        """
        self.interleaver.set_order(array)
        self.deinterleaver.set_order(array)

    def forward(self, inputs):
        x_sys = inputs[:, :, 0].view((inputs.size()[0], inputs.size()[1], 1))
        x_p1 = inputs[:, :, 1].view((inputs.size()[0], inputs.size()[1], 1))
        if self.args["rate"] == "onethird":
            x_p2 = inputs[:, :, 2].view((inputs.size()[0], inputs.size()[1], 1))

        x_inp = torch.cat([x_sys, x_p1], dim=2)

        x = x_inp.view(x_inp.size()[0], 1, x_inp.size()[1], x_inp.size()[2])
        x = self._cnn1_2d(x)
        x = self._bn_1_2d(x)
        x = func.relu(x, inplace=True)

        x = self._layers_2d(x)

        x = self._cnn_2_2d(x)
        x = x.squeeze(1)

        if self.args["rate"] == "onethird":
            x_p2 = x_p2.permute(0, 2, 1)
            x_p2 = self._cnn_1_1d(x_p2)
            x_p2 = self._bn_1_1d(x_p2)
            x_p2 = func.relu(x_p2, inplace=True)
            x_p2 = self._layers_1d(x_p2)
            x_p2 = self._cnn_2_1d(x_p2)
            x_p2 = x_p2.permute(0, 2, 1)
            x_p2 = torch.flatten(x_p2, start_dim=1)
            x_p2 = self.actf(self._dropout(self._linear_3(x_p2)))
            x_p2 = x_p2.reshape((inputs.size()[0], self.args["block_length"], 1))

        x_sys = x[:, :, 0].view((x.size()[0], x.size()[1], 1))
        x_p1 = x[:, :, 1].view((x.size()[0], x.size()[1], 1))

        x_sys = torch.flatten(x_sys, start_dim=1)
        x_sys = self.actf(self._dropout(self._linear_1(x_sys)))
        x_sys = x_sys.reshape((inputs.size()[0], self.args["block_length"], 1))

        x_p1 = torch.flatten(x_p1, start_dim=1)
        x_p1 = self.actf(self._dropout(self._linear_2(x_p1)))
        x_p1 = x_p1.reshape((inputs.size()[0], self.args["block_length"], 1))

        if self.args["rate"] == "onethird":
            x = torch.cat([x_sys, x_p1, x_p2], dim=2)
        else:
            x = torch.cat([x_sys, x_p1], dim=2)
        if not self.args["channel"] == "continuous" or self.args["continuous_coder"]:
            x = Quantizer.apply(x)

        return x

class ResNetCoder_sep(CoderBase):
    def __init__(self, arguments):
        super(ResNetCoder_sep, self).__init__(arguments)

        self._dropout = torch.nn.Dropout(self.args["coder_dropout"])

        self._linear_1 = torch.nn.Linear((self.args["block_length"] + self.args["block_padding"]
                                          + self.args["redundancy"]), self.args["block_length"]) #+16

        self.conv1 = torch.nn.Conv1d((self.args["block_length"] + self.args["block_padding"] + self.args["redundancy"]),
                                     self.args["coder_units"], kernel_size=self.args["coder_kernel"], padding=self.args["coder_kernel"]//2) #3,1
        self.bn1 = torch.nn.BatchNorm1d(self.args["coder_units"])

        layers = []
        for i in range(self.args["coder_layers"]):
            layers.append(ResNetBlock(self.args["coder_units"], self.args["coder_units"], kernel_size=self.args["coder_kernel"], padding=self.args["coder_kernel"]//2)) #5,2
        self.layers = torch.nn.Sequential(*layers)

        self.conv2 = torch.nn.Conv1d(self.args["coder_units"], (self.args["block_length"] + self.args["block_padding"]
                                                                + self.args["redundancy"]), kernel_size=self.args["coder_kernel"], padding=self.args["coder_kernel"]//2) #3,1

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = func.relu(x, inplace=True)
        x = self.layers(x)
        x = self.conv2(x)

        x = torch.flatten(x, start_dim=1)
        x = self.actf(self._dropout(self._linear_1(x)))
        x = x.reshape((inputs.size()[0], self.args["block_length"], 1))
        if not self.args["channel"] == "continuous" or self.args["continuous_coder"]:
            x = Quantizer.apply(x)

        return x

class ResNetCoder_conc(CoderBase):
    def __init__(self, arguments):
        super(ResNetCoder_conc, self).__init__(arguments)

        self._dropout = torch.nn.Dropout(0)

        self._dropout = torch.nn.Dropout(self.args["coder_dropout"])

        self._linear = torch.nn.Linear((self.args["block_length"] + self.args["block_padding"]
                                        + self.args["redundancy"])*3, self.args["block_length"]*3)


        self._cnn_1 = torch.nn.Conv1d(1, self.args["coder_units"], kernel_size=self.args["coder_kernel"], padding=self.args["coder_kernel"]//2) #27,13
        self._bn_1 = torch.nn.BatchNorm1d(self.args["coder_units"]) #out_channels

        layers = []
        for i in range(self.args["coder_layers"]):
            layers.append(ResNetBlock(self.args["coder_units"], self.args["coder_units"], kernel_size=self.args["coder_kernel"], padding=self.args["coder_kernel"]//2)) #21, 10
        self._layers = torch.nn.Sequential(*layers)

        '''
        self._cnn2 = Conv1d(self.args["coder_actf"],
                           layers=self.args["coder_layers"],
                           in_channels=self.args["coder_units"],
                           out_channels=1,
                           kernel_size=15) #self.args["coder_kernel"]
        '''
        self._cnn_2 = torch.nn.Conv1d(self.args["coder_units"], 1, kernel_size=self.args["coder_kernel"], padding=self.args["coder_kernel"]//2)

    def forward(self, inputs):
        x = inputs.transpose(1, 2).reshape(self.args["batch_size"], -1, 1).transpose(1, 2)

        x = self._cnn_1(x)
        x = self._bn_1(x)
        x = func.relu(x, inplace=True)

        x = self._layers(x)

        x = self._cnn_2(x)

        x = torch.flatten(x, start_dim=1)
        x = self.actf(self._dropout(self._linear(x)))
        x = x.reshape((inputs.size()[0], self.args["block_length"], 3))

        if not self.args["channel"] == "continuous" or self.args["continuous_coder"]:
            x = Quantizer.apply(x)

        return x

class CNN_sep(CoderBase):
    def __init__(self, arguments):
        super(CNN_sep, self).__init__(arguments)

        self._dropout = torch.nn.Dropout(self.args["coder_dropout"])

        self._cnn_1 = Conv1d(self.args["coder_actf"],
                             layers=self.args["coder_layers"],
                             in_channels=1,
                             out_channels=self.args["coder_units"],
                             kernel_size=self.args["coder_kernel"])
        self._linear_1 = torch.nn.Linear(
            self.args["coder_units"] * (self.args["block_length"] + self.args["block_padding"]
                                        + self.args["redundancy"]), self.args["block_length"])


    def forward(self, inputs):

        x = self._cnn_1(inputs)
        x = torch.flatten(x, start_dim=1)
        x = self.actf(self._dropout(self._linear_1(x)))
        x = x.reshape((inputs.size()[0], self.args["block_length"], 1))
        if not self.args["channel"] == "continuous" or self.args["continuous_coder"]:
            x = Quantizer.apply(x)

        return x

class CoderCNN_3linears(CoderBase):
    def __init__(self, arguments):
        """
        CNN based network. Used to reconstruct the code stream from the encoder after
        applying the noisy channel.
        :param arguments: Arguments as dictionary.
        """
        super(CoderCNN_3linears, self).__init__(arguments)

        self._dropout = torch.nn.Dropout(self.args["coder_dropout"])

        self._cnn_1 = Conv1d(self.args["coder_actf"],
                             layers=self.args["coder_layers"],
                             in_channels=1,
                             out_channels=self.args["coder_units"],
                             kernel_size=self.args["coder_kernel"])
        self._linear_1_1 = torch.nn.Linear(
            self.args["coder_units"] * (self.args["block_length"] + self.args["block_padding"] + self.args["redundancy"]),
            self.args["coder_units"] * (self.args["block_length"] + self.args["block_padding"] + self.args["redundancy"]))

        self._linear_1_2 = torch.nn.Linear(self.args["coder_units"] * (
                self.args["block_length"] + self.args["block_padding"] + self.args["redundancy"]),
                                           self.args["block_length"])
        self._linear_1_3 = torch.nn.Linear(self.args["block_length"], self.args["block_length"])

        self._cnn_2 = Conv1d(self.args["coder_actf"],
                             layers=self.args["coder_layers"],
                             in_channels=1,
                             out_channels=self.args["coder_units"],
                             kernel_size=self.args["coder_kernel"])

        self._linear_2_1 = torch.nn.Linear(
            self.args["coder_units"] * (self.args["block_length"] + self.args["block_padding"] + self.args["redundancy"]),
            self.args["coder_units"] * (self.args["block_length"] + self.args["block_padding"] + self.args["redundancy"]))

        self._linear_2_2 = torch.nn.Linear(self.args["coder_units"] * (
                self.args["block_length"] + self.args["block_padding"] + self.args["redundancy"]),
                                           self.args["block_length"])
        self._linear_2_3 = torch.nn.Linear(self.args["block_length"], self.args["block_length"])

        if self.args["rate"] == "onethird":
            self._cnn_3 = Conv1d(self.args["coder_actf"],
                                 layers=self.args["coder_layers"],
                                 in_channels=1,
                                 out_channels=self.args["coder_units"],
                                 kernel_size=self.args["coder_kernel"])

            self._linear_3_1 = torch.nn.Linear(
                self.args["coder_units"] * (self.args["block_length"] + self.args["block_padding"] + self.args["redundancy"]),
                self.args["coder_units"] * (self.args["block_length"] + self.args["block_padding"] + self.args["redundancy"]))

            self._linear_3_2 = torch.nn.Linear(
                self.args["coder_units"] * (self.args["block_length"] + self.args["block_padding"] + self.args["redundancy"]),
                self.args["block_length"])
            self._linear_3_3 = torch.nn.Linear(self.args["block_length"], self.args["block_length"])


    def set_parallel(self):
        """
        Ensures that forward and backward propagation operations can be performed on multiple GPUs.
        """
        self._cnn_1 = torch.nn.DataParallel(self._cnn_1)
        self._cnn_2 = torch.nn.DataParallel(self._cnn_2)
        #self._linear_1 = torch.nn.DataParallel(self._linear_1)
        #self._linear_2 = torch.nn.DataParallel(self._linear_2)
        if self.args["rate"] == "onethird":
            self._cnn_3 = torch.nn.DataParallel(self._cnn_3)
            #self._linear_3 = torch.nn.DataParallel(self._linear_3)

    def forward(self, inputs):
        """
        Calculates output tensors from input tensors based on the process.
        :param inputs: Input tensor.
        :return: Output tensor of coder.
        """
        x_sys = inputs[:, :, 0].view((inputs.size()[0], inputs.size()[1], 1))
        x_sys = self._cnn_1(x_sys)
        x_sys = torch.flatten(x_sys, start_dim=1)
        x_sys = self.actf(self._dropout(self._linear_1_1(x_sys)))
        x_sys = self.actf(self._dropout(self._linear_1_2(x_sys)))
        x_sys = self.actf(self._dropout(self._linear_1_3(x_sys)))
        x_sys = x_sys.reshape((inputs.size()[0], self.args["block_length"], 1))

        x_p1 = inputs[:, :, 1].view((inputs.size()[0], inputs.size()[1], 1))
        x_p1 = self._cnn_2(x_p1)
        x_p1 = torch.flatten(x_p1, start_dim=1)
        x_p1 = self.actf(self._dropout(self._linear_2_1(x_p1)))
        x_p1 = self.actf(self._dropout(self._linear_2_2(x_p1)))
        x_p1 = self.actf(self._dropout(self._linear_2_3(x_p1)))
        x_p1 = x_p1.reshape((inputs.size()[0], self.args["block_length"], 1))

        if self.args["rate"] == "onethird":
            x_p2 = inputs[:, :, 2].view((inputs.size()[0], inputs.size()[1], 1))
            x_p2 = self._cnn_3(x_p2)
            x_p2 = torch.flatten(x_p2, start_dim=1)
            x_p2 = self.actf(self._dropout(self._linear_3_1(x_p2)))
            x_p2 = self.actf(self._dropout(self._linear_3_2(x_p2)))
            x_p2 = self.actf(self._dropout(self._linear_3_3(x_p2)))
            x_p2 = x_p2.reshape((inputs.size()[0], self.args["block_length"], 1))

            x = torch.cat([x_sys, x_p1, x_p2], dim=2)
        else:
            x = torch.cat([x_sys, x_p1], dim=2)
        if not self.args["channel"] == "continuous" or self.args["continuous_coder"]:
            x = Quantizer.apply(x)
        return x

