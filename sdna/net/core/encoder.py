# -*- coding: utf-8 -*-

from sdna.net.core.interleaver import *
from sdna.net.core.layers import *


class EncoderBase(torch.nn.Module):
    def __init__(self, arguments):
        """
        Class serves as a encoder template, it provides utility functions.

        :param arguments: Arguments as dictionary.
        """
        super(EncoderBase, self).__init__()
        self.args = arguments

    def set_interleaver_order(self, array):
        """
        Inheritance function to set the models interleaver/de-interleaver order.

        :param array: That array that is needed to set/restore interleaver order.
        """
        pass

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
        if self.args["enc_actf"] == "tanh":
            return torch.tanh(inputs)
        elif self.args["enc_actf"] == "elu":
            return func.elu(inputs)
        elif self.args["enc_actf"] == "relu":
            return torch.relu(inputs)
        elif self.args["enc_actf"] == "selu":
            return func.selu(inputs)
        elif self.args["enc_actf"] == "sigmoid":
            return torch.sigmoid(inputs)
        elif self.args["enc_actf"] == "identity":
            return inputs
        else:
            return inputs

    @staticmethod
    def normalize(inputs):
        """
        Normalize and quantize a tensor from the encoder.

        :param inputs: Input tensor.
        :return: Normalized & quantized output tensor.
        """
        x_norm = (inputs - torch.mean(inputs)) * 1.0 / torch.std(inputs)
        x = Quantizer.apply(x_norm)
        return x


# RNN Encoder with interleaver
class EncoderRNN(EncoderBase):
    def __init__(self, arguments):
        """
        RNN based encoder with an interleaver.

        :param arguments: Arguments as dictionary.
        """
        super(EncoderRNN, self).__init__(arguments)

        self._interleaver = Interleaver()

        rnn = torch.nn.RNN
        if self.args["enc_rnn"].lower() == 'gru':
            rnn = torch.nn.GRU
        elif self.args["enc_rnn"].lower() == 'lstm':
            rnn = torch.nn.LSTM

        self._dropout = torch.nn.Dropout(self.args["enc_dropout"])

        self._rnn_1 = rnn(1, self.args["enc_units"],
                          num_layers=self.args["enc_layers"],
                          bias=True,
                          batch_first=True,
                          dropout=0,
                          bidirectional=True)
        self._linear_1 = torch.nn.Linear(2 * self.args["enc_units"], 1)
        self._rnn_2 = rnn(1, self.args["enc_units"],
                          num_layers=self.args["enc_layers"],
                          bias=True,
                          batch_first=True,
                          dropout=0,
                          bidirectional=True)
        self._linear_2 = torch.nn.Linear(2 * self.args["enc_units"], 1)

        if self.args["rate"] == "onethird":
            self._rnn_3 = rnn(1, self.args["enc_units"],
                              num_layers=self.args["enc_layers"],
                              bias=True,
                              batch_first=True,
                              dropout=0,
                              bidirectional=True)
            self._linear_3 = torch.nn.Linear(2 * self.args["enc_units"], 1)

    def set_interleaver_order(self, array):
        """
        Inheritance function to set the models interleaver order.

        :param array: That array that is needed to set interleaver order.
        """
        self._interleaver.set_order(array)

    def set_parallel(self):
        """
        Ensures that forward and backward propagation operations can be performed on multiple GPUs.
        """
        self._rnn_1 = torch.nn.DataParallel(self._rnn_1)
        self._rnn_2 = torch.nn.DataParallel(self._rnn_2)
        self._linear_1 = torch.nn.DataParallel(self._linear_1)
        self._linear_2 = torch.nn.DataParallel(self._linear_2)
        if self.args["rate"]:
            self._rnn_3 = torch.nn.DataParallel(self._rnn_3)
            self._linear_3 = torch.nn.DataParallel(self._linear_3)

    def forward(self, inputs):
        """
        Calculates output tensors from input tensors based on the process.

        :param inputs: Input tensor.
        :return: Output tensor of encoder.
        """
        inputs = 2.0 * inputs - 1.0

        x_sys, _ = self._rnn_1(inputs)
        x_sys = self.actf(self._dropout(self._linear_1(x_sys)))

        if self.args["rate"] == "onethird":
            x_p1, _ = self._rnn_2(inputs)
            x_p1 = self.actf(self._dropout(self._linear_2(x_p1)))

            x_inter = self._interleaver(inputs)
            x_p2, _ = self._rnn_3(x_inter)
            x_p2 = self.actf(self._dropout(self._linear_3(x_p2)))

            x_o = torch.cat([x_sys, x_p1, x_p2], dim=2)
        else:
            x_inter = self._interleaver(inputs)
            x_p1, _ = self._rnn_2(x_inter)
            x_p1 = self.actf(self._dropout(self._linear_2(x_p1)))

            x_o = torch.cat([x_sys, x_p1], dim=2)

        x = EncoderBase.normalize(x_o)
        return x


# RNN Encoder with systematic code, interleaver
class SysEncoderRNN(EncoderBase):
    def __init__(self, arguments):
        """
        RNN based systematic encoder with an interleaver.

        :param arguments: Arguments as dictionary.
        """
        super(SysEncoderRNN, self).__init__(arguments)

        self._interleaver = Interleaver()

        rnn = torch.nn.RNN
        if self.args["enc_rnn"].lower() == 'gru':
            rnn = torch.nn.GRU
        elif self.args["enc_rnn"].lower() == 'lstm':
            rnn = torch.nn.LSTM

        self._dropout = torch.nn.Dropout(self.args["enc_dropout"])

        self._rnn_1 = rnn(1, self.args["enc_units"],
                          num_layers=self.args["enc_layers"],
                          bias=True,
                          batch_first=True,
                          dropout=0,
                          bidirectional=True)
        self._linear_1 = torch.nn.Linear(2 * self.args["enc_units"], 1)

        if self.args["rate"] == "onethird":
            self._rnn_2 = rnn(1, self.args["enc_units"],
                              num_layers=self.args["enc_layers"],
                              bias=True,
                              batch_first=True,
                              dropout=0,
                              bidirectional=True)
            self._linear_2 = torch.nn.Linear(2 * self.args["enc_units"], 1)

    def set_interleaver_order(self, array):
        """
        Inheritance function to set the models interleaver order.

        :param array: That array that is needed to set interleaver order.
        """
        self._interleaver.set_order(array)

    def set_parallel(self):
        """
        Ensures that forward and backward propagation operations can be performed on multiple GPUs.
        """
        self._rnn_1 = torch.nn.DataParallel(self._rnn_1)
        self._linear_1 = torch.nn.DataParallel(self._linear_1)
        if self.args["rate"] == "onethird":
            self._rnn_2 = torch.nn.DataParallel(self._rnn_2)
            self._linear_2 = torch.nn.DataParallel(self._linear_2)

    def forward(self, inputs):
        """
        Calculates output tensors from input tensors based on the process.

        :param inputs: Input tensor.
        :return: Output tensor of encoder.
        """
        inputs = 2.0 * inputs - 1.0

        if self.args["rate"] == "onethird":
            x_p1, _ = self._rnn_1(inputs)
            x_p1 = self.actf(self._dropout(self._linear_1(x_p1)))

            x_inter = self._interleaver(inputs)
            x_p2, _ = self._rnn_2(x_inter)
            x_p2 = self.actf(self._dropout(self._linear_2(x_p2)))

            x_t = EncoderBase.normalize(torch.cat([x_p1, x_p2], dim=2))
        else:
            x_inter = self._interleaver(inputs)
            x_p1, _ = self._rnn_1(x_inter)
            x_p1 = self.actf(self._dropout(self._linear_1(x_p1)))
            x_t = EncoderBase.normalize(x_p1)
        x = torch.cat([inputs, x_t], dim=2)
        return x


# CNN Encoder with interleaver
class EncoderCNN(EncoderBase):
    def __init__(self, arguments):
        """
        CNN based encoder with an interleaver.

        :param arguments: Arguments as dictionary.
        """
        super(EncoderCNN, self).__init__(arguments)

        self._interleaver = Interleaver()

        self._dropout = torch.nn.Dropout(self.args["enc_dropout"])

        self._cnn_1 = Conv1d(self.args["enc_actf"],
                             layers=self.args["enc_layers"],
                             in_channels=1,
                             out_channels=self.args["enc_units"],
                             kernel_size=self.args["enc_kernel"])
        self._latent_1 = Conv1d(self.args["enc_actf"],
                             layers=1,
                             in_channels=self.args["enc_units"],
                             out_channels=self.args["enc_units"], #+16,  # ToDo: make it a parameter
                             kernel_size=self.args["enc_kernel"])
        self._linear_1 = torch.nn.Linear(self.args["enc_units"], 1) #+16
        self._cnn_2 = Conv1d(self.args["enc_actf"],
                             layers=self.args["enc_layers"],
                             in_channels=1,
                             out_channels=self.args["enc_units"],
                             kernel_size=self.args["enc_kernel"])
        self._latent_2 = Conv1d(self.args["enc_actf"],
                             layers=1,
                             in_channels=self.args["enc_units"],
                             out_channels=self.args["enc_units"], #+16
                             kernel_size=self.args["enc_kernel"])
        self._linear_2 = torch.nn.Linear(self.args["enc_units"], 1) #+16

        if self.args["rate"] == "onethird":
            self._cnn_3 = Conv1d(self.args["enc_actf"],
                                 layers=self.args["enc_layers"],
                                 in_channels=1,
                                 out_channels=self.args["enc_units"],
                                 kernel_size=self.args["enc_kernel"])
            self._latent_3 = Conv1d(self.args["enc_actf"],
                                    layers=1,
                                    in_channels=self.args["enc_units"],
                                    out_channels=self.args["enc_units"], #+16
                                    kernel_size=self.args["enc_kernel"])
            self._linear_3 = torch.nn.Linear(self.args["enc_units"], 1) #+16

    def set_interleaver_order(self, array):
        """
        Inheritance function to set the models interleaver order.

        :param array: That array that is needed to set interleaver order.
        """
        self._interleaver.set_order(array)

    def set_parallel(self):
        """
        Ensures that forward and backward propagation operations can be performed on multiple GPUs.
        """
        self._cnn_1 = torch.nn.DataParallel(self._cnn_1)
        self._cnn_2 = torch.nn.DataParallel(self._cnn_2)
        self._latent_1 = torch.nn.DataParallel(self._latent_1)
        self._latent_2 = torch.nn.DataParallel(self._latent_2)
        self._linear_1 = torch.nn.DataParallel(self._linear_1)
        self._linear_2 = torch.nn.DataParallel(self._linear_2)
        if self.args["rate"] == "onethird":
            self._cnn_3 = torch.nn.DataParallel(self._cnn_3)
            self._latent_3 = torch.nn.DataParallel(self._latent_3)
            self._linear_3 = torch.nn.DataParallel(self._linear_3)

    def forward(self, inputs):
        """
        Calculates output tensors from input tensors based on the process.

        :param inputs: Input tensor.
        :return: Output tensor of encoder.
        """
        inputs = 2.0 * inputs - 1.0

        x_sys = self._cnn_1(inputs)
        x_sys = self._latent_1(x_sys)
        x_sys = self.actf(self._dropout(self._linear_1(x_sys)))

        if self.args["rate"] == "onethird":
            x_p1 = self._cnn_2(inputs)
            x_p1 = self._latent_2(x_p1)
            x_p1 = self.actf(self._dropout(self._linear_2(x_p1)))

            x_inter = self._interleaver(inputs)
            x_p2 = self._cnn_3(x_inter)
            x_p2 = self._latent_3(x_p2)
            x_p2 = self.actf(self._dropout(self._linear_3(x_p2)))

            x_o = torch.cat([x_sys, x_p1, x_p2], dim=2)
        else:
            x_inter = self._interleaver(inputs)
            x_p1 = self._cnn_2(x_inter)
            x_p1 = self._latent_2(x_p1)
            x_p1 = self.actf(self._dropout(self._linear_2(x_p1)))
            x_o = torch.cat([x_sys, x_p1], dim=2)

        x = EncoderBase.normalize(x_o)
        return x


# CNN Encoder with systematic code, interleaver
class SysEncoderCNN(EncoderBase):
    def __init__(self, arguments):
        """
        CNN based systematic encoder with an interleaver.

        :param arguments: Arguments as dictionary.
        """
        super(SysEncoderCNN, self).__init__(arguments)

        self._interleaver = Interleaver()

        self._dropout = torch.nn.Dropout(self.args["enc_dropout"])

        self._cnn_1 = Conv1d(self.args["enc_actf"],
                             layers=self.args["enc_layers"],
                             in_channels=1,
                             out_channels=self.args["enc_units"],
                             kernel_size=self.args["enc_kernel"])
        self._linear_1 = torch.nn.Linear(self.args["enc_units"], 1)
        if self.args["rate"] == "onethird":
            self._cnn_2 = Conv1d(self.args["enc_actf"],
                                 layers=self.args["enc_layers"],
                                 in_channels=1,
                                 out_channels=self.args["enc_units"],
                                 kernel_size=self.args["enc_kernel"])
            self._linear_2 = torch.nn.Linear(self.args["enc_units"], 1)

    def set_interleaver_order(self, array):
        """
        Inheritance function to set the models interleaver order.

        :param array: That array that is needed to set interleaver order.
        """
        self._interleaver.set_order(array)

    def set_parallel(self):
        """
        Ensures that forward and backward propagation operations can be performed on multiple GPUs.
        """
        self._cnn_1 = torch.nn.DataParallel(self._cnn_1)
        self._linear_1 = torch.nn.DataParallel(self._linear_1)
        if self.args["rate"] == "onethird":
            self._cnn_2 = torch.nn.DataParallel(self._cnn_2)
            self._linear_2 = torch.nn.DataParallel(self._linear_2)

    def forward(self, inputs):
        """
        Calculates output tensors from input tensors based on the process.

        :param inputs: Input tensor.
        :return: Output tensor of encoder.
        """
        inputs = 2.0 * inputs - 1.0

        if self.args["rate"] == "onethird":
            x_p1 = self._cnn_1(inputs)
            x_p1 = self.actf(self._dropout(self._linear_1(x_p1)))

            x_inter = self._interleaver(inputs)
            x_p2 = self._cnn_2(x_inter)
            x_p2 = self.actf(self._dropout(self._linear_2(x_p2)))

            x_t = EncoderBase.normalize(torch.cat([x_p1, x_p2], dim=2))
        else:
            x_inter = self._interleaver(inputs)
            x_p1 = self._cnn_1(x_inter)
            x_p1 = self.actf(self._dropout(self._linear_1(x_p1)))

            x_t = EncoderBase.normalize(x_p1)
        x = torch.cat([inputs, x_t], dim=2)
        return x


class EncoderRNNatt(EncoderBase):
    def __init__(self, arguments):
        """
        RNN based encoder with a code rate of 1/3 and an interleaver.

        :param arguments: Arguments as dictionary.
        """
        super(EncoderRNNatt, self).__init__(arguments)

        self._interleaver = Interleaver()

        rnn = torch.nn.RNN
        if self.args["enc_rnn"].lower() == 'gru':
            rnn = torch.nn.GRU
        elif self.args["enc_rnn"].lower() == 'lstm':
            rnn = torch.nn.LSTM

        self._dropout = torch.nn.Dropout(self.args["enc_dropout"])

        self._rnn_1 = rnn(1, self.args["enc_units"],
                          num_layers=self.args["enc_layers"],
                          bias=True,
                          batch_first=True,
                          dropout=0,
                          bidirectional=True)
        self._linear_1 = torch.nn.Linear(2 * self.args["enc_units"], 1)
        self._rnn_2 = rnn(1, self.args["enc_units"],
                          num_layers=self.args["enc_layers"],
                          bias=True,
                          batch_first=True,
                          dropout=0,
                          bidirectional=True)
        self._linear_2 = torch.nn.Linear(2 * self.args["enc_units"], 1)
        if self.args["rate"] == "onethird":
            self._rnn_3 = rnn(1, self.args["enc_units"],
                              num_layers=self.args["enc_layers"],
                              bias=True,
                              batch_first=True,
                              dropout=0,
                              bidirectional=True)
            self._linear_3 = torch.nn.Linear(2 * self.args["enc_units"], 1)

    def set_interleaver_order(self, array):
        """
        Inheritance function to set the models interleaver order.

        :param array: That array that is needed to set interleaver order.
        """
        self._interleaver.set_order(array)

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

    def forward(self, inputs, hidden):
        """
        Calculates output tensors from input tensors based on the process.

        :param inputs: Input tensor.
        :return: Output tensor of encoder.
        """
        inputs = 2.0 * inputs - 1.0

        # ToDo check if the hidden vector has to be added to the "x_sys".
        x_sys, _ = self._rnn_1(inputs)
        x_sys = self.actf(self._dropout(self._linear_1(x_sys)))

        if self.args["rate"] == "onethird":
            x_p1, hidden[0] = self._rnn_2(inputs, hidden[0])
            x_p1 = self.actf(self._dropout(self._linear_2(x_p1)))

            x_inter = self._interleaver(inputs)
            x_p2, hidden[1] = self._rnn_3(x_inter, hidden[1])
            x_p2 = self.actf(self._dropout(self._linear_3(x_p2)))

            x_o = torch.cat([x_sys, x_p1, x_p2], dim=2)
        else:
            x_inter = self._interleaver(inputs)
            x_p1, hidden[0] = self._rnn_2(x_inter, hidden[0])
            x_p1 = self.actf(self._dropout(self._linear_2(x_p1)))
            x_o = torch.cat([x_sys, x_p1], dim=2)
        x = EncoderBase.normalize(x_o)
        return x, hidden

    def initHidden(self):
        return 2 * [torch.zeros(2 * self.args["enc_layers"], self.args["batch_size"], self.args[
            "enc_units"])]  # [torch.zeros(10, 256, self.args["enc_units"]), torch.zeros(10, 256, self.args["enc_units"]), torch.zeros(10, 256, self.args["enc_units"])]


class EncoderTransformer(EncoderBase):
    def __init__(self, arguments):
        """
        Transformer based encoder with an interleaver.

        :param arguments: Arguments as dictionary.
        """
        super(EncoderTransformer, self).__init__(arguments)

        self._interleaver = Interleaver()

        self._dropout = torch.nn.Dropout(self.args["enc_dropout"])

        #Dropout and activation function might have to be removed, as they are used further down.
        #Also, torch might have to be updated, as it doesnt know batch first.
        print(self.args["enc_kernel"])
        self._encoder_layer_1 = torch.nn.TransformerEncoderLayer(d_model=self.args["enc_units"],
                                                                 nhead=self.args["enc_kernel"],
                                                                 dropout=self.args["enc_dropout"],
                                                                 activation='relu', #only relu or gelu work as activation function
                                                                 batch_first=True)
        self._transformer_1 = torch.nn.TransformerEncoder(self._encoder_layer_1, num_layers=self.args["enc_layers"])
        self._linear_1 = torch.nn.Linear(1, self.args["enc_units"])

        self._encoder_layer_2 = torch.nn.TransformerEncoderLayer(d_model=self.args["enc_units"],
                                                                 nhead=self.args["enc_kernel"],
                                                                 dropout=self.args["enc_dropout"],
                                                                 activation='relu', #only relu or gelu work as activation function
                                                                 batch_first=True)
        self._transformer_2 = torch.nn.TransformerEncoder(self._encoder_layer_2, num_layers=self.args["enc_layers"])
        self._linear_2 = torch.nn.Linear(1, self.args["enc_units"])

        if self.args["rate"] == "onethird":
            self._encoder_layer_3 = torch.nn.TransformerEncoderLayer(d_model=self.args["enc_units"],
                                                                     nhead=self.args["enc_kernel"],
                                                                     dropout=self.args["enc_dropout"],
                                                                     activation='relu', #only relu or gelu work as activation function
                                                                     batch_first=True)
            self._transformer_3 = torch.nn.TransformerEncoder(self._encoder_layer_3, num_layers=self.args["enc_layers"])
            self._linear_3 = torch.nn.Linear(1, self.args["enc_units"])

    def set_interleaver_order(self, array):
        """
        Inheritance function to set the models interleaver order.

        :param array: That array that is needed to set interleaver order.
        """
        self._interleaver.set_order(array)

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
        :return: Output tensor of encoder.
        """
        #inputs = 2.0 * inputs - 1.0
        #inputs = inputs.view(self.args["batch_size"], self.args["block_length"], -1)

        x_sys = self._linear_1(inputs)
        x_sys = self._transformer_1(x_sys)
        x_sys = self.actf(self._dropout(x_sys))

        if self.args["rate"] == "onethird":
            x_p1 = self._linear_2(inputs)
            x_p1 = self._transformer_2(x_p1)
            x_p1 = self.actf(self._dropout(x_p1))

            x_inter = self._interleaver(inputs)
            x_inter = self._linear_3(x_inter)
            x_p2 = self._transformer_3(x_inter)
            x_p2 = self.actf(self._dropout(x_p2))

            x_o = torch.cat([x_sys, x_p1, x_p2], dim=2)
        else:
            x_inter = self._linear_2(inputs)
            x_inter = self._interleaver(x_inter)
            x_p1 = self._transformer_2(x_inter)
            x_p1 = self.actf(self._dropout(x_p1))

            x_o = torch.cat([x_sys, x_p1], dim=2)

        x = EncoderBase.normalize(x_o)
        return x

        """
        x_inter = self._interleaver(inputs)
        x_inter = self._transformer_2(x_inter)
        x_inter = self.actf(self._dropout(x_inter))

        if self.args["rate"] == "onethird":
            x_inter_2 = self._interleaver(x_inter)
            x_inter_2 = self._transformer_3(x_inter_2)
            x_inter_2 = self.actf(self._dropout(x_inter_2))
            x_o = torch.cat([x_sys, x_inter, x_inter_2], dim=2)
        else:
            x_o = torch.cat([x_sys, x_inter], dim=2)
        return x_o
        """