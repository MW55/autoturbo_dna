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


class CoderTransformer(CoderBase):
    def __init__(self, arguments):
        """
        Transformer based network. Used to reconstruct the code stream from the encoder after
        applying the noisy channel.

        :param arguments: Arguments as dictionary.
        """
        super(CoderTransformer, self).__init__(arguments)

        self.transformer = transformers.Transformer(
            d_model=self.args["d_model"],
            nhead=self.args["nhead"],
            num_layers=self.args["num_layers"],
            dim_feedforward=self.args["dim_feedforward"],
            dropout=self.args["coder_dropout"],
        )

    def set_parallel(self):
        """
        Ensures that forward and backward propagation operations can be performed on multiple GPUs.
        """
        self.transformer = torch.nn.DataParallel(self.transformer)

    def forward(self, inputs):
        """
        Calculates output tensors from input tensors based on the process.

        :param inputs: Input tensor.
        :return: Output tensor of coder.
        """
        x_sys = inputs[:, :, 0].view((inputs.size()[0], inputs.size()[1]))
        x_sys = self.transformer(x_sys)
        x_sys = self.actf(x_sys)
        x_sys = x_sys.reshape((inputs.size()[0], self.args["block_length"], 1))

        x_p1 = inputs[:, :, 1].view((inputs.size()[0], inputs.size()[1]))
        x_p1 = self.transformer(x_p1)
        x_p1 = self.actf(x_p1)
        x_p1 = x_p1.reshape((inputs.size()[0], self.args["block_length"], 1))

        if self.args["rate"] == "onethird":
            x_p2 = inputs[:, :, 2].view((inputs.size()[0], inputs.size()[1]))
            x_p2 = self.transformer(x_p2)
            x_p2 = self.actf(x_p2)
            x_p2 = x_p2.reshape((inputs.size()[0], self.args["block_length"], 1))
            return torch.cat((x_sys, x_p1, x_p2), dim=2)

        return torch.cat((x_sys, x_p1), dim=2)