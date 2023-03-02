# -*- coding: utf-8 -*-

from sdna.net.core.layers import *


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

        self._linear_1 = torch.nn.Linear(self.args["block_length"] + self.args["block_padding"], self.args["block_length"])
        self._linear_2 = torch.nn.Linear(self.args["block_length"] + self.args["block_padding"], self.args["block_length"])
        if self.args["rate"] == "onethird":
            self._linear_3 = torch.nn.Linear(self.args["block_length"] + self.args["block_padding"], self.args["block_length"])

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
        self._linear_1 = torch.nn.Linear((self.args["coder_units"]) * (self.args["block_length"] + int(self.args["redundancy"]) + self.args["block_padding"]), self.args["block_length"]+int(self.args["redundancy"])) #+16
        self._cnn_2 = Conv1d(self.args["coder_actf"],
                             layers=self.args["coder_layers"],
                             in_channels=1,
                             out_channels=self.args["coder_units"], #+16
                             kernel_size=self.args["coder_kernel"])
        self._linear_2 = torch.nn.Linear((self.args["coder_units"]) * (self.args["block_length"] + int(self.args["redundancy"]) + self.args["block_padding"]), self.args["block_length"]+int(self.args["redundancy"])) #+16
        self._batch_norm_1 = torch.nn.BatchNorm1d(self.args["block_length"] + int(self.args["redundancy"]))
        self._batch_norm_2 = torch.nn.BatchNorm1d(self.args["block_length"] + int(self.args["redundancy"]))
        if self.args["rate"] == "onethird":
            self._cnn_3 = Conv1d(self.args["coder_actf"],
                                 layers=self.args["coder_layers"],
                                 in_channels=1,
                                 out_channels=self.args["coder_units"], #+16
                                 kernel_size=self.args["coder_kernel"])
            self._linear_3 = torch.nn.Linear((self.args["coder_units"]) * (self.args["block_length"] + int(self.args["redundancy"]) + self.args["block_padding"]), self.args["block_length"]+int(self.args["redundancy"])) #+16
            self._batch_norm_3 = torch.nn.BatchNorm1d(self.args["block_length"] + int(self.args["redundancy"]))

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

        x_sys = x_sys.reshape((inputs.size()[0], self.args["block_length"]+int(self.args["redundancy"]), 1))
        x_sys = self._batch_norm_1(x_sys)

        x_p1 = inputs[:, :, 1].view((inputs.size()[0], inputs.size()[1], 1))
        x_p1 = self._cnn_2(x_p1)
        x_p1 = torch.flatten(x_p1, start_dim=1)
        x_p1 = self.actf(self._dropout(self._linear_2(x_p1)))
        x_p1 = x_p1.reshape((inputs.size()[0], self.args["block_length"]+int(self.args["redundancy"]), 1))
        x_p1 = self._batch_norm_2(x_p1)

        if self.args["rate"] == "onethird":
            x_p2 = inputs[:, :, 2].view((inputs.size()[0], inputs.size()[1], 1))
            x_p2 = self._cnn_3(x_p2)
            x_p2 = torch.flatten(x_p2, start_dim=1)
            x_p2 = self.actf(self._dropout(self._linear_3(x_p2)))
            x_p2 = x_p2.reshape((inputs.size()[0], self.args["block_length"]+int(self.args["redundancy"]), 1))
            x_p2 = self._batch_norm_3(x_p2)

            x = torch.cat([x_sys, x_p1, x_p2], dim=2)
        else:
            x = torch.cat([x_sys, x_p1], dim=2)
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
        self._linear_1 = torch.nn.Linear(2 * self.args["coder_units"] * (self.args["block_length"] + self.args["block_padding"]), self.args["block_length"])
        self._rnn_2 = rnn(1, self.args["coder_units"],
                          num_layers=self.args["coder_layers"],
                          bias=True,
                          batch_first=True,
                          dropout=0,
                          bidirectional=True)
        self._linear_2 = torch.nn.Linear(2 * self.args["coder_units"] * (self.args["block_length"] + self.args["block_padding"]), self.args["block_length"])

        if self.args["rate"] == "onethird":
            self._rnn_3 = rnn(1, self.args["coder_units"],
                              num_layers=self.args["coder_layers"],
                              bias=True,
                              batch_first=True,
                              dropout=0,
                              bidirectional=True)
            self._linear_3 = torch.nn.Linear(2 * self.args["coder_units"] * (self.args["block_length"] + self.args["block_padding"]), self.args["block_length"])

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
        x = Quantizer.apply(x)
        return x

'''
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
        self._cnn_2 = Conv1d(self.args["coder_actf"],
                             layers=self.args["coder_layers"],
                             in_channels=1,
                             out_channels=self.args["coder_units"],
                             kernel_size=self.args["coder_kernel"])
        self._linear_2 = torch.nn.Linear(self.args["coder_units"] * (self.args["block_length"] + self.args["block_padding"]), self.args["block_length"])
        if self.args["rate"] == "onethird":
            self._cnn_3 = Conv1d(self.args["coder_actf"],
                                 layers=self.args["coder_layers"],
                                 in_channels=1,
                                 out_channels=self.args["coder_units"],
                                 kernel_size=self.args["coder_kernel"])
            self._linear_3 = torch.nn.Linear(self.args["coder_units"] * (self.args["block_length"] + self.args["block_padding"]), self.args["block_length"])

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
        x = Quantizer.apply(x)
        return x
'''
# CNN Coder
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
        #self._linear_1 = torch.nn.Linear(self.args["coder_units"] * (self.args["block_length"] + self.args["block_padding"]), self.args["block_length"])
        self._linear_1 = torch.nn.Linear(self.args["coder_units"] * (self.args["block_length"]), self.args["block_length"])
        self._cnn_2 = Conv1d(self.args["coder_actf"],
                             layers=self.args["coder_layers"],
                             in_channels=1,
                             out_channels=self.args["coder_units"],
                             kernel_size=self.args["coder_kernel"])
        #self._linear_2 = torch.nn.Linear(self.args["coder_units"] * (self.args["block_length"] + self.args["block_padding"]), self.args["block_length"])
        self._linear_2 = torch.nn.Linear(
            self.args["coder_units"] * (self.args["block_length"]),
            self.args["block_length"])
        if self.args["rate"] == "onethird":
            self._cnn_3 = Conv1d(self.args["coder_actf"],
                                 layers=self.args["coder_layers"],
                                 in_channels=1,
                                 out_channels=self.args["coder_units"],
                                 kernel_size=self.args["coder_kernel"])
            #self._linear_3 = torch.nn.Linear(self.args["coder_units"] * (self.args["block_length"] + self.args["block_padding"]), self.args["block_length"])
            self._linear_3 = torch.nn.Linear(
                self.args["coder_units"] * (self.args["block_length"]),
                self.args["block_length"])

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
        self._linear_1 = torch.nn.Linear(self.args["coder_units"] * (self.args["block_length"] + self.args["block_padding"]), self.args["block_length"])
        self._cnn_2 = Conv1d(self.args["coder_actf"],
                             layers=self.args["coder_layers"],
                             in_channels=1,
                             out_channels=self.args["coder_units"],
                             kernel_size=self.args["coder_kernel"])
        self._rnn_2 = torch.nn.GRU(input_size=self.args["coder_units"],
                                   hidden_size=self.args["coder_units"],
                                   num_layers=self.args["coder_layers"],
                                   batch_first=True)
        self._linear_2 = torch.nn.Linear(self.args["coder_units"] * (self.args["block_length"] + self.args["block_padding"]), self.args["block_length"])
        if self.args["batch_norm"]:
            self._batch_norm_1 = torch.nn.BatchNorm1d(self.args["block_length"] + self.args["block_padding"])
            self._batch_norm_2 = torch.nn.BatchNorm1d(self.args["block_length"] + self.args["block_padding"])
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
            self._linear_3 = torch.nn.Linear(self.args["coder_units"] * (self.args["block_length"] + self.args["block_padding"]), self.args["block_length"])
            if self.args["batch_norm"]:
                self._batch_norm_3 = torch.nn.BatchNorm1d(self.args["block_length"] + self.args["block_padding"])
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

        #Dropout and activation function might have to be removed, as they are used further down.
        #Also, torch might have to be updated, as it doesnt know batch first.
        self._encoder_layer_1 = torch.nn.TransformerEncoderLayer(d_model=self.args["coder_units"],
                                                                 nhead=self.args["coder_kernel"],
                                                                 dropout=self.args["coder_dropout"],
                                                                 activation='relu', #only relu or gelu work as activation function
                                                                 batch_first=True)
        self._transformer_1 = torch.nn.TransformerEncoder(self._encoder_layer_1, num_layers=self.args["coder_layers"])
        self._linear_1 = torch.nn.Linear(
            self.args["coder_units"] * (self.args["block_length"] + self.args["block_padding"]),
            self.args["block_length"])

        self._encoder_layer_2 = torch.nn.TransformerEncoderLayer(d_model=self.args["coder_units"],
                                                                 nhead=self.args["coder_kernel"],
                                                                 dropout=self.args["coder_dropout"],
                                                                 activation='relu', #only relu or gelu work as activation function
                                                                 batch_first=True)
        self._transformer_2 = torch.nn.TransformerEncoder(self._encoder_layer_2, num_layers=self.args["coder_layers"])
        self._linear_2 = torch.nn.Linear(self.args["coder_units"] * (self.args["block_length"] + self.args["block_padding"]), self.args["block_length"])

        if self.args["rate"] == "onethird":
            self._encoder_layer_3 = torch.nn.TransformerEncoderLayer(d_model=self.args["coder_units"],
                                                                     nhead=self.args["coder_kernel"],
                                                                     dropout=self.args["coder_dropout"],
                                                                     activation='relu', #only relu or gelu work as activation function
                                                                     batch_first=True)
            self._transformer_3 = torch.nn.TransformerEncoder(self._encoder_layer_3, num_layers=self.args["coder_layers"])
            self._linear_3 = torch.nn.Linear(self.args["coder_units"] * (self.args["block_length"] + self.args["block_padding"]), self.args["block_length"])

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
        self._linear = torch.nn.Linear(self.args["coder_units"] * 3 * (self.args["block_length"] + self.args["block_padding"]), self.args["block_length"]*3)

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
        #x_sys = inputs[:, :, 0].view((inputs.size()[0], inputs.size()[1], 1))
        #x_p1 = inputs[:, :, 1].view((inputs.size()[0], inputs.size()[1], 1))
        #x_p2 = inputs[:, :, 2].view((inputs.size()[0], inputs.size()[1], 1))
        x = inputs.transpose(1, 2).reshape(self.args["batch_size"], -1, 1)
        #concatenated = torch.cat([x_sys, x_p1, x_p2], dim=1)
        #concatenated = concatenated.reshape((inputs.size()[0], (self.args["block_length"]+self.args["block_padding"])*3, 1))

        #x = inputs.view((inputs.size()[0], inputs.size()[1] * 3, 1))
        x = self._cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.actf(self._dropout(self._linear(x)))
        x = x.reshape((inputs.size()[0], self.args["block_length"], 3))
        x = Quantizer.apply(x)
        return x