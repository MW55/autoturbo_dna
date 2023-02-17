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
        elif self.args["enc_actf"] == "leakyrelu":
            return func.leaky_relu(inputs)
        elif self.args["enc_actf"] == "gelu":
            return func.gelu(inputs)
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
        '''
        self._latent_1_1 = Conv1d(self.args["enc_actf"],
                             layers=3,
                             in_channels=self.args["block_length"],
                             out_channels=self.args["block_length"] + int(self.args["redundancy"]/2), #4, #+16,  # ToDo: make it a parameter
                             kernel_size=self.args["enc_kernel"])
        self._latent_1_2 = Conv1d(self.args["enc_actf"],
                             layers=3,
                             in_channels=self.args["block_length"] + int(self.args["redundancy"]/2),
                             out_channels=self.args["block_length"] + int(self.args["redundancy"]),
                             kernel_size=self.args["enc_kernel"])
        '''
        self._latent_1_1 = torch.nn.Linear(self.args["block_length"],  # + 16
                                              self.args["block_length"] + int(self.args["redundancy"]/2))
        self._latent_1_2 = torch.nn.Linear(self.args["block_length"] + int(self.args["redundancy"]/2),
                                           self.args["block_length"] + int(self.args["redundancy"]))
        self._latent_2_1 = torch.nn.Linear(self.args["block_length"],  # + 16
                                              self.args["block_length"] + int(self.args["redundancy"]/2))
        self._latent_2_2 = torch.nn.Linear(self.args["block_length"] + int(self.args["redundancy"]/2),
                                                self.args["block_length"] + int(self.args["redundancy"]))
        if self.args["batch_norm"]:
            self._batch_norm_1 = torch.nn.BatchNorm1d(self.args["block_length"] + int(self.args["redundancy"]))
            self._batch_norm_2 = torch.nn.BatchNorm1d(self.args["block_length"] + int(self.args["redundancy"]))


        self._linear_1 = torch.nn.Linear(self.args["enc_units"], 1) #+16
        self._cnn_2 = Conv1d(self.args["enc_actf"],
                             layers=self.args["enc_layers"],
                             in_channels=1,
                             out_channels=self.args["enc_units"],
                             kernel_size=self.args["enc_kernel"])
        '''
        self._latent_2_1 = Conv1d(self.args["enc_actf"],
                             layers=1,
                             in_channels=self.args["block_length"],
                             out_channels=self.args["block_length"] + int(self.args["redundancy"]/2),
                             kernel_size=self.args["enc_kernel"])
        self._latent_2_2 = Conv1d(self.args["enc_actf"],
                             layers=1,
                             in_channels=self.args["block_length"] + int(self.args["redundancy"]/2),
                             out_channels=self.args["block_length"] + int(self.args["redundancy"]),
                             kernel_size=self.args["enc_kernel"])
        '''
        self._linear_2 = torch.nn.Linear(self.args["enc_units"], 1) #+16

        if self.args["rate"] == "onethird":
            self._cnn_3 = Conv1d(self.args["enc_actf"],
                                 layers=self.args["enc_layers"],
                                 in_channels=1,
                                 out_channels=self.args["enc_units"],
                                 kernel_size=self.args["enc_kernel"])
            '''
            self._latent_3_1 = Conv1d(self.args["enc_actf"],
                                    layers=1,
                                    in_channels=self.args["block_length"],
                                    out_channels=self.args["block_length"] + int(self.args["redundancy"]/2),
                                    kernel_size=self.args["enc_kernel"])
            self._latent_3_2 = Conv1d(self.args["enc_actf"],
                                    layers=1,
                                    in_channels=self.args["block_length"] + int(self.args["redundancy"]/2),
                                    out_channels=self.args["block_length"] + int(self.args["redundancy"]),
                                    kernel_size=self.args["enc_kernel"])
            '''
            self._latent_3_1 = torch.nn.Linear(self.args["block_length"],  # + 16
                                               self.args["block_length"] + int(self.args["redundancy"] / 2))
            self._latent_3_2 = torch.nn.Linear(self.args["block_length"] + int(self.args["redundancy"] / 2),
                                               self.args["block_length"] + int(self.args["redundancy"]))
            if self.args["batch_norm"]:
                self._batch_norm_3 = torch.nn.BatchNorm1d(self.args["block_length"] + int(self.args["redundancy"]))
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
        self._latent_1_1 = torch.nn.DataParallel(self._latent_1_1)
        self._latent_1_2 = torch.nn.DataParallel(self._latent_1_2)
        self._latent_2_1 = torch.nn.DataParallel(self._latent_2_1)
        self._latent_2_2 = torch.nn.DataParallel(self._latent_2_2)
        self._linear_1 = torch.nn.DataParallel(self._linear_1)
        self._linear_2 = torch.nn.DataParallel(self._linear_2)
        if self.args["batch_norm"]:
            self._batch_norm_1 = torch.nn.DataParallel(self._batch_norm_1)
            self._batch_norm_2 = torch.nn.DataParallel(self._batch_norm_2)
        if self.args["rate"] == "onethird":
            self._cnn_3 = torch.nn.DataParallel(self._cnn_3)
            self._latent_3_1 = torch.nn.DataParallel(self._latent_3_1)
            self._latent_3_1 = torch.nn.DataParallel(self._latent_3_1)
            self._linear_3 = torch.nn.DataParallel(self._linear_3)
            if self.args["batch_norm"]:
                self._batch_norm_3 = torch.nn.DataParallel(self._batch_norm_3)

    def forward(self, inputs):
        """
        Calculates output tensors from input tensors based on the process.

        :param inputs: Input tensor.
        :return: Output tensor of encoder.
        """
        inputs = 2.0 * inputs - 1.0

        x_sys = self._cnn_1(inputs)
        x_sys = self.actf(self._dropout(self._linear_1(x_sys)))

        ###test
        x_sys = torch.flatten(x_sys, start_dim=1)
        #x_sys = x_sys.reshape((inputs.size()[0], self.args["block_length"]+int(self.args["redundancy"]), 1))
        x_sys = self._latent_1_1(x_sys)
        x_sys = self._latent_1_2(x_sys)
        #x_sys = self.actf(self._dropout(self._latent_1_1(x_sys)))
        #x_sys = self.actf(self._dropout(self._latent_1_2(x_sys)))
        x_sys = x_sys.reshape((inputs.size()[0], self.args["block_length"]+int(self.args["redundancy"]), 1))
        ###test over

        #x_sys = x_sys.permute(0, 2, 1)
        #x_sys = self._latent_1_1(x_sys)
        #x_sys = self.actf(self._dropout(x_sys)) #new
        #x_sys = self._latent_1_2(x_sys)
        #x_sys = self.actf(self._dropout(x_sys)) #new
        #x_sys = x_sys.permute(0, 2, 1)
        #x_sys = self._batch_norm_1(x_sys) #ToDo: try batch normalization after the activation function
        #x_sys = self._linear_1(x_sys)
        #x_sys = self.actf(self._dropout(x_sys))
        if self.args["batch_norm"]:
            x_sys = self._batch_norm_1(x_sys)

        if self.args["rate"] == "onethird":
            x_p1 = self._cnn_2(inputs)
            x_p1 = self.actf(self._dropout(self._linear_2(x_p1)))
            ##test
            x_p1 = torch.flatten(x_p1, start_dim=1)
            #x_p1 = x_p1.reshape((inputs.size()[0], self.args["block_length"] + int(self.args["redundancy"]), 1))
            x_p1 = self._latent_2_1(x_p1)
            x_p1 = self._latent_2_2(x_p1)
            #x_p1 = self.actf(self._dropout(self._latent_2_1(x_p1)))
            #x_p1 = self.actf(self._dropout(self._latent_2_2(x_p1)))
            x_p1 = x_p1.reshape((inputs.size()[0], self.args["block_length"] + int(self.args["redundancy"]), 1))
            ###test over
            #x_p1 = x_p1.permute(0, 2, 1)
            #x_p1 = self._latent_2_1(x_p1)
            #x_p1 = self.actf(self._dropout(x_p1)) #new, could dropout lead to different sized outputs?
            #x_p1 = self._latent_2_2(x_p1)
            #x_p1 = self.actf(self._dropout(x_p1)) #new, could dropout lead to different sized outputs?
            #x_p1 = x_p1.permute(0, 2, 1)
            #x_p1 = self._batch_norm_2(x_p1)
            #x_p1 = self._linear_2(x_p1)
            #x_p1 = self.actf(self._dropout(x_p1))
            if self.args["batch_norm"]:
                x_p1 = self._batch_norm_2(x_p1)

            x_inter = self._interleaver(inputs)
            x_p2 = self._cnn_3(x_inter)
            x_p2 = self.actf(self._dropout(self._linear_3(x_p2)))
            ###test
            x_p2 = torch.flatten(x_p2, start_dim=1)
            #x_p2 = x_p2.reshape((inputs.size()[0], self.args["block_length"]+int(self.args["redundancy"]), 1))
            x_p2 = self._latent_3_1(x_p2)
            x_p2 = self._latent_3_2(x_p2)
            #x_p2 = self.actf(self._dropout(self._latent_3_1(x_p2)))
            #x_p2 = self.actf(self._dropout(self._latent_3_2(x_p2)))
            x_p2 = x_p2.reshape((inputs.size()[0], self.args["block_length"]+int(self.args["redundancy"]), 1))
            ###test over
            #x_p2 = x_p2.permute(0, 2, 1)
            #x_p2 = self._latent_3_1(x_p2)
            #x_p2 = self.actf(self._dropout(x_p2)) #new, could dropout lead to different sized outputs?
            #x_p2 = self._latent_3_2(x_p2)
            #x_p2 = self.actf(self._dropout(x_p2)) #new, could dropout lead to different sized outputs?
            #x_p2 = x_p2.permute(0, 2, 1)
            #x_p2 = self._batch_norm_3(x_p2)
            #x_p2 = self._linear_3(x_p2)
            #x_p2 = self.actf(self._dropout(x_p2))
            if self.args["batch_norm"]:
                x_p2 = self._batch_norm_3(x_p2)

            x_o = torch.cat([x_sys, x_p1, x_p2], dim=2)
        else:
            x_inter = self._interleaver(inputs)
            x_p1 = self._cnn_2(x_inter)
            x_p1 = self.actf(self._dropout(self._linear_2(x_p1)))
            ###test
            x_p1 = torch.flatten(x_p1, start_dim=1)
            #x_p1 = x_p1.reshape((inputs.size()[0], self.args["block_length"] + int(self.args["redundancy"]), 1))
            x_p1 = self.actf(self._dropout(self._latent_2_1(x_p1)))
            x_p1 = self.actf(self._dropout(self._latent_2_2(x_p1)))
            x_p1 = x_p1.reshape((inputs.size()[0], self.args["block_length"] + int(self.args["redundancy"]), 1))
            ###test over
            #x_p1 = x_p1.permute(0, 2, 1)
            #x_p1 = self._latent_2_1(x_p1)
            #x_p1 = self.actf(self._dropout(x_p1)) #new, could dropout lead to different sized outputs?
            #x_p1 = self._latent_2_2(x_p1)
            #x_p1 = self.actf(self._dropout(x_p1)) #new, could dropout lead to different sized outputs?
            #x_p1 = x_p1.permute(0, 2, 1)
            #x_p1 = self._batch_norm_2(x_p1)
            #x_p1 = self._linear_2(x_p1)
            #x_p1 = self.actf(self._dropout(x_p1))
            if self.args["batch_norm"]:
                x_p1 = self._batch_norm_2(x_p1)

            x_o = torch.cat([x_sys, x_p1], dim=2)

        x = EncoderBase.normalize(x_o)
        return x

# CNN Encoder with interleaver
class EncoderCNN_nolat(EncoderBase):
    def __init__(self, arguments):
        """
        CNN based encoder with an interleaver.
        :param arguments: Arguments as dictionary.
        """
        super(EncoderCNN_nolat, self).__init__(arguments)

        self._interleaver = Interleaver()

        self._dropout = torch.nn.Dropout(self.args["enc_dropout"])

        self._cnn_1 = Conv1d(self.args["enc_actf"],
                             layers=self.args["enc_layers"],
                             in_channels=1,
                             out_channels=self.args["enc_units"],
                             kernel_size=self.args["enc_kernel"])
        self._linear_1 = torch.nn.Linear(self.args["enc_units"], 1)
        self._cnn_2 = Conv1d(self.args["enc_actf"],
                             layers=self.args["enc_layers"],
                             in_channels=1,
                             out_channels=self.args["enc_units"],
                             kernel_size=self.args["enc_kernel"])
        self._linear_2 = torch.nn.Linear(self.args["enc_units"], 1)

        if self.args["batch_norm"]:
            self._batch_norm_1 = torch.nn.BatchNorm1d(self.args["block_length"])
            self._batch_norm_2 = torch.nn.BatchNorm1d(self.args["block_length"])


        if self.args["rate"] == "onethird":
            self._cnn_3 = Conv1d(self.args["enc_actf"],
                                 layers=self.args["enc_layers"],
                                 in_channels=1,
                                 out_channels=self.args["enc_units"],
                                 kernel_size=self.args["enc_kernel"])
            self._linear_3 = torch.nn.Linear(self.args["enc_units"], 1)

            if self.args["batch_norm"]:
                self._batch_norm_3 = torch.nn.BatchNorm1d(self.args["block_length"])


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
        self._linear_1 = torch.nn.DataParallel(self._linear_1)
        self._linear_2 = torch.nn.DataParallel(self._linear_2)
        if self.args["batch_norm"]:
            self._batch_norm_1 = torch.nn.DataParallel(self._batch_norm_1)
            self._batch_norm_2 = torch.nn.DataParallel(self._batch_norm_2)
        if self.args["rate"] == "onethird":
            self._cnn_3 = torch.nn.DataParallel(self._cnn_3)
            self._linear_3 = torch.nn.DataParallel(self._linear_3)
            if self.args["batch_norm"]:
                self._batch_norm_3 = torch.nn.DataParallel(self._batch_norm_3)

    def forward(self, inputs):
        """
        Calculates output tensors from input tensors based on the process.
        :param inputs: Input tensor.
        :return: Output tensor of encoder.
        """
        inputs = 2.0 * inputs - 1.0

        x_sys = self._cnn_1(inputs)
        x_sys = self.actf(self._dropout(self._linear_1(x_sys)))
        if self.args["batch_norm"]:
            x_sys = self._batch_norm_1(x_sys)

        if self.args["rate"] == "onethird":
            x_p1 = self._cnn_2(inputs)
            x_p1 = self.actf(self._dropout(self._linear_2(x_p1)))
            if self.args["batch_norm"]:
                x_p1 = self._batch_norm_2(x_p1)

            x_inter = self._interleaver(inputs)
            x_p2 = self._cnn_3(x_inter)
            x_p2 = self.actf(self._dropout(self._linear_3(x_p2)))
            if self.args["batch_norm"]:
                x_p2 = self._batch_norm_3(x_p2)

            x_o = torch.cat([x_sys, x_p1, x_p2], dim=2)
        else:
            x_inter = self._interleaver(inputs)
            x_p1 = self._cnn_2(x_inter)
            x_p1 = self.actf(self._dropout(self._linear_2(x_p1)))
            x_o = torch.cat([x_sys, x_p1], dim=2)
            if self.args["batch_norm"]:
                x_p1 = self._batch_norm_2(x_p1)

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

class Encoder_vae(EncoderBase):
    def __init__(self, arguments):
        """
        CNN based encoder with an interleaver.
        :param arguments: Arguments as dictionary.
        """
        super(Encoder_vae, self).__init__(arguments)

        self._interleaver = Interleaver()

        self._dropout = torch.nn.Dropout(self.args["enc_dropout"])

        self._cnn_1 = Conv1d(self.args["enc_actf"],
                             layers=self.args["enc_layers"],
                             in_channels=1,
                             out_channels=self.args["enc_units"],
                             kernel_size=self.args["enc_kernel"])
        self._linear_1_1 = torch.nn.Linear(self.args["enc_units"], 1)
        self._linear_1_2 = torch.nn.Linear(self.args["enc_units"], 1)
        self._linear_1_3 = torch.nn.Linear(self.args["enc_units"], 1)
        self._cnn_2 = Conv1d(self.args["enc_actf"],
                             layers=self.args["enc_layers"],
                             in_channels=1,
                             out_channels=self.args["enc_units"],
                             kernel_size=self.args["enc_kernel"])
        self._linear_2_1 = torch.nn.Linear(self.args["enc_units"], 1)
        self._linear_2_2 = torch.nn.Linear(self.args["enc_units"], 1)
        self._linear_2_3 = torch.nn.Linear(self.args["enc_units"], 1)

        if self.args["batch_norm"]:
            self._batch_norm_1 = torch.nn.BatchNorm1d(self.args["block_length"] + int(self.args["redundancy"]))
            self._batch_norm_2 = torch.nn.BatchNorm1d(self.args["block_length"] + int(self.args["redundancy"]))


        if self.args["rate"] == "onethird":
            self._cnn_3 = Conv1d(self.args["enc_actf"],
                                 layers=self.args["enc_layers"],
                                 in_channels=1,
                                 out_channels=self.args["enc_units"],
                                 kernel_size=self.args["enc_kernel"])
            self._linear_3_1 = torch.nn.Linear(self.args["enc_units"], 1)
            self._linear_3_2 = torch.nn.Linear(self.args["enc_units"], 1)
            self._linear_3_3 = torch.nn.Linear(self.args["enc_units"], 1)
            if self.args["batch_norm"]:
                self._batch_norm_3 = torch.nn.BatchNorm1d(self.args["block_length"] + int(self.args["redundancy"]))


        #self.N_1 = torch.distributions.Normal(0, 1)
        self.N_1 = torch.distributions.Bernoulli(probs=torch.tensor(0.5))
        #self.N_1.loc = self.N_1.loc()  # self.N.loc.cuda() hack to get sampling on the GP
        #self.N_1.scale = self.N_1.scale() #self.N.scale.cuda()
        self.kl_1 = 0

        #self.N_2 = torch.distributions.Normal(0, 1)
        self.N_2 = torch.distributions.Bernoulli(probs=torch.tensor(0.5))
        #self.N_2.loc = self.N_2.loc()  # hack to get sampling on the GPU
        #self.N_2.scale = self.N_2.scale()
        self.kl_2 = 0

        if self.args["rate"] == "onethird":
            #self.N_3 = torch.distributions.Normal(0, 1)
            self.N_3 = torch.distributions.Bernoulli(probs=torch.tensor(0.5))
            #self.N_3.loc = self.N_3.loc()  # hack to get sampling on the GPU
            #self.N_3.scale = self.N_3.scale()
            self.kl_3 = 0

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
        self._linear_1_1 = torch.nn.DataParallel(self._linear_1_1)
        self._linear_1_2 = torch.nn.DataParallel(self._linear_1_2)
        self._linear_1_3 = torch.nn.DataParallel(self._linear_1_3)
        self._linear_2_1 = torch.nn.DataParallel(self._linear_2_1)
        self._linear_2_2 = torch.nn.DataParallel(self._linear_2_2)
        self._linear_2_3 = torch.nn.DataParallel(self._linear_2_3)
        if self.args["batch_norm"]:
            self._batch_norm_1 =  torch.nn.DataParallel(self._batch_norm_1)
            self._batch_norm_2 = torch.nn.DataParallel(self._batch_norm_2)
        if self.args["rate"] == "onethird":
            self._cnn_3 = torch.nn.DataParallel(self._cnn_3)
            self._linear_3_1 = torch.nn.DataParallel(self._linear_3_1)
            self._linear_3_2 = torch.nn.DataParallel(self._linear_3_2)
            self._linear_3_3 = torch.nn.DataParallel(self._linear_3_3)
            if self.args["batch_norm"]:
                self._batch_norm_3 = torch.nn.DataParallel(self._batch_norm_3)

    def forward(self, inputs):
        """
        Calculates output tensors from input tensors based on the process.
        :param inputs: Input tensor.
        :return: Output tensor of encoder.
        """
        inputs = 2.0 * inputs - 1.0

        x_sys = self._cnn_1(inputs)
        #x_sys = self.actf(self._dropout(self._linear_1_1(x_sys)))
        mu_1 = self._linear_1_2(x_sys)
        sigma_1 = torch.exp(self._linear_1_3(x_sys))
        x_sys_z = mu_1 + sigma_1*self.N_1.sample(mu_1.shape)
        self.kl_1 = (sigma_1**2 + mu_1**2 - torch.log(sigma_1) - 1/2).sum()
        if self.args["batch_norm"]:
            x_sys_z = self._batch_norm_1(x_sys_z)

        if self.args["rate"] == "onethird":
            x_p1 = self._cnn_2(inputs)
            #x_p1 = self.actf(self._dropout(self._linear_2_1(x_p1)))
            mu_2 = self._linear_2_2(x_p1)
            sigma_2 = torch.exp(self._linear_2_3(x_p1))
            x_p1_z = mu_2 + sigma_2 * self.N_2.sample(mu_2.shape)
            self.kl_2 = (sigma_2 ** 2 + mu_2 ** 2 - torch.log(sigma_2) - 1 / 2).sum()
            if self.args["batch_norm"]:
                x_p1_z = self._batch_norm_2(x_p1_z)

            x_inter = self._interleaver(inputs)
            x_p2 = self._cnn_3(x_inter)
            #x_p2 = self.actf(self._dropout(self._linear_3_1(x_p2)))
            mu_3 = self._linear_3_2(x_p2)
            sigma_3 = torch.exp(self._linear_3_3(x_p2))
            x_p2_z = mu_3 + sigma_3 * self.N_3.sample(mu_3.shape)
            self.kl_3 = (sigma_3 ** 2 + mu_3 ** 2 - torch.log(sigma_3) - 1 / 2).sum()
            if self.args["batch_norm"]:
                x_p2_z = self._batch_norm_3(x_p2_z)

            x_o = torch.cat([x_sys_z, x_p1_z, x_p2_z], dim=2)
        else:
            x_inter = self._interleaver(inputs)
            x_p1 = self._cnn_2(x_inter)
            #x_p1 = self.actf(self._dropout(self._linear_2_1(x_p1)))
            mu_2 = self._linear2_2(x_p1)
            sigma_2 = torch.exp(self.linear_2_3(x_p1))

            x_p1_z = mu_2 + sigma_2 * self.N_2.sample(mu_2.shape)
            self.kl_2 = (sigma_2 ** 2 + mu_2 ** 2 - torch.log(sigma_2) - 1 / 2).sum()
            if self.args["batch_norm"]:
                x_p1_z = self._batch_norm_2(x_p1_z)

            x_o = torch.cat([x_sys_z, x_p1_z], dim=2)

        x = EncoderBase.normalize(x_o)
        return x

class Encoder_vae_lat(EncoderBase):
    def __init__(self, arguments):
        """
        CNN based encoder with an interleaver.
        :param arguments: Arguments as dictionary.
        """
        super(Encoder_vae_lat, self).__init__(arguments)

        self._interleaver = Interleaver()

        self._dropout = torch.nn.Dropout(self.args["enc_dropout"])

        self._cnn_1 = Conv1d(self.args["enc_actf"],
                             layers=self.args["enc_layers"],
                             in_channels=1,
                             out_channels=self.args["enc_units"],
                             kernel_size=self.args["enc_kernel"])
        self._linear_1_1 = torch.nn.Linear(self.args["enc_units"], 1)
        self._linear_1_2 = torch.nn.Linear(self.args["enc_units"], 1)
        self._linear_1_3 = torch.nn.Linear(self.args["enc_units"], 1)
        self._cnn_2 = Conv1d(self.args["enc_actf"],
                             layers=self.args["enc_layers"],
                             in_channels=1,
                             out_channels=self.args["enc_units"],
                             kernel_size=self.args["enc_kernel"])
        self._linear_2_1 = torch.nn.Linear(self.args["enc_units"], 1)
        self._linear_2_2 = torch.nn.Linear(self.args["enc_units"], 1)
        self._linear_2_3 = torch.nn.Linear(self.args["enc_units"], 1)

        if self.args["batch_norm"]:
            self._batch_norm_1 = torch.nn.BatchNorm1d(self.args["block_length"] + int(self.args["redundancy"]))
            self._batch_norm_2 = torch.nn.BatchNorm1d(self.args["block_length"] + int(self.args["redundancy"]))

        if self.args["rate"] == "onethird":
            self._cnn_3 = Conv1d(self.args["enc_actf"],
                                 layers=self.args["enc_layers"],
                                 in_channels=1,
                                 out_channels=self.args["enc_units"],
                                 kernel_size=self.args["enc_kernel"])
            self._linear_3_1 = torch.nn.Linear(self.args["enc_units"], 1)
            self._linear_3_2 = torch.nn.Linear(self.args["enc_units"], 1)
            self._linear_3_3 = torch.nn.Linear(self.args["enc_units"], 1)
            self._latent_3_1 = torch.nn.Linear(self.args["block_length"],  # + 16
                                               self.args["block_length"] + int(self.args["redundancy"] / 2))
            self._latent_3_2 = torch.nn.Linear(self.args["block_length"] + int(self.args["redundancy"] / 2),
                                               self.args["block_length"] + int(self.args["redundancy"]))
            if self.args["batch_norm"]:
                self._batch_norm_3 = torch.nn.BatchNorm1d(self.args["block_length"] + int(self.args["redundancy"]))

        self._latent_1_1 = torch.nn.Linear(self.args["block_length"],  # + 16
                                              self.args["block_length"] + int(self.args["redundancy"]/2))
        self._latent_1_2 = torch.nn.Linear(self.args["block_length"] + int(self.args["redundancy"]/2),
                                           self.args["block_length"] + int(self.args["redundancy"]))
        self._latent_2_1 = torch.nn.Linear(self.args["block_length"],  # + 16
                                              self.args["block_length"] + int(self.args["redundancy"]/2))
        self._latent_2_2 = torch.nn.Linear(self.args["block_length"] + int(self.args["redundancy"]/2),
                                                self.args["block_length"] + int(self.args["redundancy"]))
        # self.N_1 = torch.distributions.Normal(0, 1)
        self.N_1 = torch.distributions.Bernoulli(probs=torch.tensor(0.5))
        # self.N_1.loc = self.N_1.loc()  # self.N.loc.cuda() hack to get sampling on the GP
        # self.N_1.scale = self.N_1.scale() #self.N.scale.cuda()
        self.kl_1 = 0

        # self.N_2 = torch.distributions.Normal(0, 1)
        self.N_2 = torch.distributions.Bernoulli(probs=torch.tensor(0.5))
        # self.N_2.loc = self.N_2.loc()  # hack to get sampling on the GPU
        # self.N_2.scale = self.N_2.scale()
        self.kl_2 = 0

        if self.args["rate"] == "onethird":
            # self.N_3 = torch.distributions.Normal(0, 1)
            self.N_3 = torch.distributions.Bernoulli(probs=torch.tensor(0.5))
            # self.N_3.loc = self.N_3.loc()  # hack to get sampling on the GPU
            # self.N_3.scale = self.N_3.scale()
            self.kl_3 = 0

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
        self._latent_1_1 = torch.nn.DataParallel(self._latent_1_1)
        self._latent_1_2 = torch.nn.DataParallel(self._latent_1_2)
        self._latent_2_1 = torch.nn.DataParallel(self._latent_2_1)
        self._latent_2_2 = torch.nn.DataParallel(self._latent_2_2)
        self._linear_1_1 = torch.nn.DataParallel(self._linear_1_1)
        self._linear_1_2 = torch.nn.DataParallel(self._linear_1_2)
        self._linear_1_3 = torch.nn.DataParallel(self._linear_1_3)
        self._linear_2_1 = torch.nn.DataParallel(self._linear_2_1)
        self._linear_2_2 = torch.nn.DataParallel(self._linear_2_2)
        self._linear_2_3 = torch.nn.DataParallel(self._linear_2_3)
        if self.args["batch_norm"]:
            self._batch_norm_1 = torch.nn.DataParallel(self._batch_norm_1)
            self._batch_norm_2 = torch.nn.DataParallel(self._batch_norm_2)
        if self.args["rate"] == "onethird":
            self._cnn_3 = torch.nn.DataParallel(self._cnn_3)
            self._latent_3_1 = torch.nn.DataParallel(self._latent_3_1)
            self._latent_3_1 = torch.nn.DataParallel(self._latent_3_1)
            self._linear_3_1 = torch.nn.DataParallel(self._linear_3_1)
            self._linear_3_2 = torch.nn.DataParallel(self._linear_3_2)
            self._linear_3_3 = torch.nn.DataParallel(self._linear_3_3)
            if self.args["batch_norm"]:
                self._batch_norm_3 = torch.nn.DataParallel(self._batch_norm_3)

    def forward(self, inputs):
        """
        Calculates output tensors from input tensors based on the process.
        :param inputs: Input tensor.
        :return: Output tensor of encoder.
        """
        inputs = 2.0 * inputs - 1.0

        x_sys = self._cnn_1(inputs)

        #x_sys = torch.flatten(x_sys, start_dim=1)
        #x_sys = self._latent_1_1(x_sys)
        #x_sys = self._latent_1_2(x_sys)
        #x_sys = x_sys.reshape((inputs.size()[0], self.args["block_length"]+int(self.args["redundancy"]), 1))

        mu_1 = self._linear_1_2(x_sys)
        sigma_1 = torch.exp(self._linear_1_3(x_sys))
        x_sys_z = mu_1 + sigma_1 * self.N_1.sample(mu_1.shape)
        self.kl_1 = (sigma_1 ** 2 + mu_1 ** 2 - torch.log(sigma_1) - 1 / 2).sum()

        x_sys_z = torch.flatten(x_sys_z, start_dim=1)
        x_sys_z = self.actf(self._latent_1_1(x_sys_z))
        x_sys_z = self.actf(self._latent_1_2(x_sys_z))
        x_sys_z = x_sys_z.reshape((inputs.size()[0], self.args["block_length"] + int(self.args["redundancy"]), 1))

        if self.args["batch_norm"]:
            x_sys_z = self._batch_norm_1(x_sys_z)

        if self.args["rate"] == "onethird":
            x_p1 = self._cnn_2(inputs)
            # x_p1 = self.actf(self._dropout(self._linear_2_1(x_p1)))

            #x_p1 = torch.flatten(x_p1, start_dim=1)
            #x_p1 = self._latent_2_1(x_p1)
            #x_p1 = self._latent_2_2(x_p1)
            #x_p1 = x_p1.reshape((inputs.size()[0], self.args["block_length"] + int(self.args["redundancy"]), 1))

            mu_2 = self._linear_2_2(x_p1)
            sigma_2 = torch.exp(self._linear_2_3(x_p1))
            x_p1_z = mu_2 + sigma_2 * self.N_2.sample(mu_2.shape)
            self.kl_2 = (sigma_2 ** 2 + mu_2 ** 2 - torch.log(sigma_2) - 1 / 2).sum()

            x_p1_z = torch.flatten(x_p1_z, start_dim=1)
            x_p1_z = self.actf(self._latent_2_1(x_p1_z))
            x_p1_z = self.actf(self._latent_2_2(x_p1_z))
            x_p1_z = x_p1_z.reshape((inputs.size()[0], self.args["block_length"] + int(self.args["redundancy"]), 1))

            if self.args["batch_norm"]:
                x_p1_z = self._batch_norm_2(x_p1_z)

            x_inter = self._interleaver(inputs)
            x_p2 = self._cnn_3(x_inter)

            #x_p2 = torch.flatten(x_p2, start_dim=1)
            #x_p2 = self._latent_3_1(x_p2)
            #x_p2 = self._latent_3_2(x_p2)
            #x_p2 = x_p2.reshape((inputs.size()[0], self.args["block_length"] + int(self.args["redundancy"]), 1))

            # x_p2 = self.actf(self._dropout(self._linear_3_1(x_p2)))
            mu_3 = self._linear_3_2(x_p2)
            sigma_3 = torch.exp(self._linear_3_3(x_p2))
            x_p2_z = mu_3 + sigma_3 * self.N_3.sample(mu_3.shape)
            self.kl_3 = (sigma_3 ** 2 + mu_3 ** 2 - torch.log(sigma_3) - 1 / 2).sum()
            if self.args["batch_norm"]:
                x_p2_z = self._batch_norm_3(x_p2_z)

            x_p2_z = torch.flatten(x_p2_z, start_dim=1)
            x_p2_z = self.actf(self._latent_3_1(x_p2_z))
            x_p2_z = self.actf(self._latent_3_2(x_p2_z))
            x_p2_z = x_p2_z.reshape((inputs.size()[0], self.args["block_length"] + int(self.args["redundancy"]), 1))

            x_o = torch.cat([x_sys_z, x_p1_z, x_p2_z], dim=2)
        else:
            x_inter = self._interleaver(inputs)
            x_p1 = self._cnn_2(x_inter)

            #x_p1 = torch.flatten(x_p1, start_dim=1)
            #x_p1 = self.actf(self._dropout(self._latent_2_1(x_p1)))
            #x_p1 = self.actf(self._dropout(self._latent_2_2(x_p1)))
            #x_p1 = x_p1.reshape((inputs.size()[0], self.args["block_length"] + int(self.args["redundancy"]), 1))

            # x_p1 = self.actf(self._dropout(self._linear_2_1(x_p1)))
            mu_2 = self._linear2_2(x_p1)
            sigma_2 = torch.exp(self.linear_2_3(x_p1))

            x_p1_z = mu_2 + sigma_2 * self.N_2.sample(mu_2.shape)
            self.kl_2 = (sigma_2 ** 2 + mu_2 ** 2 - torch.log(sigma_2) - 1 / 2).sum()
            if self.args["batch_norm"]:
                x_p1_z = self._batch_norm_2(x_p1_z)

            x_p1_z = torch.flatten(x_p1_z, start_dim=1)
            x_p1_z = self._latent_2_1(x_p1_z)
            x_p1_z = self._latent_2_2(x_p1_z)
            x_p1_z = x_p1_z.reshape((inputs.size()[0], self.args["block_length"] + int(self.args["redundancy"]), 1))

            x_o = torch.cat([x_sys_z, x_p1_z], dim=2)

        x = EncoderBase.normalize(x_o)  # The normalization probabily hinders the vae to function correctly
        return x