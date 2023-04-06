# -*- coding: utf-8 -*-

from sdna.net.core.interleaver import *
from sdna.net.core.layers import *
from copy import deepcopy

class DecoderBase(torch.nn.Module):
    def __init__(self, arguments):
        """
        Class serves as a decoder template, it provides utility functions.

        :param arguments: Arguments as dictionary.
        """
        super(DecoderBase, self).__init__()
        self.is_parallel = False
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
        if self.args["dec_actf"] == "tanh":
            return torch.tanh(inputs)
        elif self.args["dec_actf"] == "elu":
            return func.elu(inputs)
        elif self.args["dec_actf"] == "relu":
            return torch.relu(inputs)
        elif self.args["enc_actf"] == "selu":
            return func.selu(inputs)
        elif self.args["dec_actf"] == "sigmoid":
            return torch.sigmoid(inputs)
        elif self.args["dec_actf"] == "identity":
            return inputs
        elif self.args["dec_actf"] == "leakyrelu":
            return func.leaky_relu(inputs)
        elif self.args["dec_actf"] == "gelu":
            return func.gelu(inputs)
        else:
            return inputs


# RNN Decoder with de/interleaver
class DecoderRNN(DecoderBase):
    def __init__(self, arguments):
        """
        RNN based decoder with an de/interleaver.

        :param arguments: Arguments as dictionary.
        """
        super(DecoderRNN, self).__init__(arguments)

        self.interleaver = Interleaver()
        self.deinterleaver = DeInterleaver()

        rnn = torch.nn.RNN
        if self.args["dec_rnn"].lower() == 'gru':
            rnn = torch.nn.GRU
        elif self.args["dec_rnn"].lower() == 'lstm':
            rnn = torch.nn.LSTM

        self._dropout = torch.nn.Dropout(self.args["dec_dropout"])

        self._rnns_1 = torch.nn.ModuleList()
        self._rnns_2 = torch.nn.ModuleList()
        self._linears_1 = torch.nn.ModuleList()
        self._linears_2 = torch.nn.ModuleList()

        for i in range(self.args["dec_iterations"]):
            self._rnns_1.append(rnn(2 + self.args["dec_inputs"], self.args["dec_units"],
                                    num_layers=self.args["dec_layers"],
                                    bias=True,
                                    batch_first=True,
                                    dropout=0,
                                    bidirectional=True))
            self._linears_1.append(torch.nn.Linear(2 * self.args["dec_units"], self.args["dec_inputs"]))
            self._rnns_2.append(rnn(2 + self.args["dec_inputs"], self.args["dec_units"],
                                    num_layers=self.args["dec_layers"],
                                    bias=True,
                                    batch_first=True,
                                    dropout=0,
                                    bidirectional=True))
            if i == self.args["dec_iterations"] - 1:
                self._linears_2.append(torch.nn.Linear(2 * self.args["dec_units"], 1))
            else:
                self._linears_2.append(torch.nn.Linear(2 * self.args["dec_units"], self.args["dec_inputs"]))

    def set_interleaver_order(self, array):
        """
        Inheritance function to set the models interleaver/de-interleaver order.

        :param array: That array that is needed to set/restore interleaver order.
        """
        self.interleaver.set_order(array)
        self.deinterleaver.set_order(array)

    def set_parallel(self):
        """
        Ensures that forward and backward propagation operations can be performed on multiple GPUs.
        """
        self.is_parallel = True
        for i in range(self.args["dec_iterations"]):
            self._rnns_1[i] = torch.nn.DataParallel(self._rnns_1[i])
            self._rnns_2[i] = torch.nn.DataParallel(self._rnns_2[i])
            self._linears_1[i] = torch.nn.DataParallel(self._linears_1[i])
            self._linears_2[i] = torch.nn.DataParallel(self._linears_2[i])

    def forward(self, inputs):
        """
        Calculates output tensors from input tensors based on the process.

        :param inputs: Input tensor.
        :return: Output tensor of decoder.
        """
        x_sys = inputs[:, :, 0].view((inputs.size()[0], inputs.size()[1], 1))
        x_sys_inter = self.interleaver(x_sys)
        x_p1 = inputs[:, :, 1].view((inputs.size()[0], inputs.size()[1], 1))
        if self.args["rate"] == "onethird":
            x_p2 = inputs[:, :, 2].view((inputs.size()[0], inputs.size()[1], 1))
        else:
            x_p1_deint = self.deinterleaver(x_p1)
        prior = torch.zeros((inputs.size()[0], inputs.size()[1], self.args["dec_inputs"]))

        if self.args["rate"] == "onethird":
            for i in range(self.args["dec_iterations"]):
                xi = torch.cat([x_sys, x_p1, prior], dim=2)
                if self.is_parallel:
                    self._rnns_1[i].module.flatten_parameters()
                x_dec, _ = self._rnns_1[i](xi)
                x = self.actf(self._dropout(self._linears_1[i](x_dec)))
                if self.args["extrinsic"]:
                    x = x - prior

                x_inter = self.interleaver(x)
                xi = torch.cat([x_sys_inter, x_p2, x_inter], dim=2)
                if self.is_parallel:
                    self._rnns_2[i].module.flatten_parameters()
                x_dec, _ = self._rnns_2[i](xi)
                x = self.actf(self._dropout(self._linears_2[i](x_dec)))
                if self.args["extrinsic"] and i != self.args["dec_iterations"] - 1:
                    x = x - x_inter

                prior = self.deinterleaver(x)
        else:
            for i in range(self.args["dec_iterations"]):
                xi = torch.cat([x_sys, x_p1_deint, prior], dim=2)
                if self.is_parallel:
                    self._rnns_1[i].module.flatten_parameters()
                x_dec, _ = self._rnns_1[i](xi)
                x = self.actf(self._dropout(self._linears_1[i](x_dec)))
                if self.args["extrinsic"]:
                    x = x - prior

                x_inter = self.interleaver(x)
                xi = torch.cat([x_sys_inter, x_p1, x_inter], dim=2)
                if self.is_parallel:
                    self._rnns_2[i].module.flatten_parameters()
                x_dec, _ = self._rnns_2[i](xi)
                x = self.actf(self._dropout(self._linears_2[i](x_dec)))
                if self.args["extrinsic"] and i != self.args["dec_iterations"] - 1:
                    x = x - x_inter

                prior = self.deinterleaver(x)

        x = torch.sigmoid(prior)
        return x


# CNN Decoder with de/interleaver
class DecoderCNN(DecoderBase):
    def __init__(self, arguments):
        """
        CNN based decoder with an de/interleaver.

        :param arguments: Arguments as dictionary.
        """
        super(DecoderCNN, self).__init__(arguments)

        self.interleaver = Interleaver()
        self.deinterleaver = DeInterleaver()

        self._dropout = torch.nn.Dropout(self.args["dec_dropout"])

        self._cnns_1 = torch.nn.ModuleList()
        self._cnns_2 = torch.nn.ModuleList()
        self._linears_1 = torch.nn.ModuleList()
        self._linears_2 = torch.nn.ModuleList()

        self._latent_1_1 = torch.nn.Linear(self.args["block_length"] + int(self.args["redundancy"]),  # + 16
                                              self.args["block_length"] + int(self.args["redundancy"]/2))
        self._latent_1_2 = torch.nn.Linear(self.args["block_length"] + int(self.args["redundancy"]/2),  # + 16
                                              self.args["block_length"])
        self._latent_2_1 = torch.nn.Linear(self.args["block_length"] + int(self.args["redundancy"]),  # + 16
                                              self.args["block_length"] + int(self.args["redundancy"]/2))
        self._latent_2_2 = torch.nn.Linear(self.args["block_length"] + int(self.args["redundancy"]/2),
                                                self.args["block_length"])
        if self.args["batch_norm"]:
            self._batch_norm_1 = torch.nn.BatchNorm1d(self.args["block_length"])

        if self.args["rate"] == "onethird":
            self._latent_3_1 = torch.nn.Linear(self.args["block_length"] + int(self.args["redundancy"]),  # + 16
                                              self.args["block_length"] + int(self.args["redundancy"]/2))
            self._latent_3_2 = torch.nn.Linear(self.args["block_length"] + int(self.args["redundancy"]/2),  # + 16
                                              self.args["block_length"])

        for i in range(self.args["dec_iterations"]):
            self._cnns_1.append(Conv1d(self.args["dec_actf"],
                                       layers=self.args["dec_layers"],
                                       in_channels=2 + self.args["dec_inputs"],
                                       out_channels=self.args["dec_units"],
                                       kernel_size=self.args["dec_kernel"]))
            self._linears_1.append(torch.nn.Linear(self.args["dec_units"], self.args["dec_inputs"]))
            self._cnns_2.append(Conv1d(self.args["dec_actf"],
                                       layers=self.args["dec_layers"],
                                       in_channels=2 + self.args["dec_inputs"],
                                       out_channels=self.args["dec_units"],
                                       kernel_size=self.args["dec_kernel"]))
            if i == self.args["dec_iterations"] - 1:
                self._linears_2.append(torch.nn.Linear(self.args["dec_units"], 1))
            else:
                self._linears_2.append(torch.nn.Linear(self.args["dec_units"], self.args["dec_inputs"]))

    def set_interleaver_order(self, array):
        """
        Inheritance function to set the models interleaver/de-interleaver order.

        :param array: That array that is needed to set/restore interleaver order.
        """
        self.interleaver.set_order(array)
        self.deinterleaver.set_order(array)

    def set_parallel(self):
        """
        Ensures that forward and backward propagation operations can be performed on multiple GPUs.
        """
        self.is_parallel = True
        self._latent_1_1 = torch.nn.DataParallel(self._latent_1_1)
        self._latent_1_2 = torch.nn.DataParallel(self._latent_1_2)
        self._latent_2_1 = torch.nn.DataParallel(self._latent_2_1)
        self._latent_2_2 = torch.nn.DataParallel(self._latent_2_2)
        if self.args["batch_norm"]:
            self._batch_norm_1 = torch.nn.DataParallel(self._batch_norm_1)
        if self.args["rate"] == "onethird":
            self._latent_3_1 = torch.nn.DataParallel(self._latent_3_1)
            self._latent_3_2 = torch.nn.DataParallel(self._latent_3_2)
        for i in range(self.args["dec_iterations"]):
            self._cnns_1[i] = torch.nn.DataParallel(self._cnns_1[i])
            self._cnns_2[i] = torch.nn.DataParallel(self._cnns_2[i])
            self._linears_1[i] = torch.nn.DataParallel(self._linears_1[i])
            self._linears_2[i] = torch.nn.DataParallel(self._linears_2[i])

    def forward(self, inputs):
        """
        Calculates output tensors from input tensors based on the process.

        :param inputs: Input tensor.
        :return: Output tensor of decoder.
        """
        x_sys = inputs[:, :, 0].view((inputs.size()[0], inputs.size()[1], 1))

        x_sys = torch.flatten(x_sys, start_dim=1)
        x_sys = self.actf(self._dropout(self._latent_1_1(x_sys)))
        x_sys = self.actf(self._dropout(self._latent_1_2(x_sys)))
        x_sys = x_sys.reshape((inputs.size()[0], self.args["block_length"], 1))

        if bool(torch.isnan(x_sys).any()):
            print("failed")

        x_sys_inter = self.interleaver(x_sys)
        x_p1 = inputs[:, :, 1].view((inputs.size()[0], inputs.size()[1], 1))

        x_p1 = torch.flatten(x_p1, start_dim=1)
        x_p1 = self.actf(self._dropout(self._latent_2_1(x_p1)))
        x_p1 = self.actf(self._dropout(self._latent_2_2(x_p1)))
        x_p1 = x_p1.reshape((inputs.size()[0], self.args["block_length"], 1))


        if bool(torch.isnan(x_p1).any()):
            print("failed")

        if self.args["rate"] == "onethird":
            x_p2 = inputs[:, :, 2].view((inputs.size()[0], inputs.size()[1], 1))
            x_p2 = torch.flatten(x_p2, start_dim=1)
            x_p2 = self.actf(self._dropout(self._latent_3_1(x_p2)))
            x_p2 = self.actf(self._dropout(self._latent_3_2(x_p2)))
            x_p2 = x_p2.reshape((inputs.size()[0], self.args["block_length"], 1))
            #test over

            if bool(torch.isnan(x_p2).any()):
                print("failed")
        else:
            x_p1_deint = self.deinterleaver(x_p1)



        prior = torch.zeros((inputs.size()[0], inputs.size()[1]-int(self.args["redundancy"]), self.args["dec_inputs"]))

        if self.args["rate"] == "onethird":
            for i in range(self.args["dec_iterations"]):
                xi = torch.cat([x_sys, x_p1, prior], dim=2)
                x_dec = self._cnns_1[i](xi)
                x = self.actf(self._dropout(self._linears_1[i](x_dec)))
                if self.args["extrinsic"]:
                    x = x - prior

                x_inter = self.interleaver(x)
                xi = torch.cat([x_sys_inter, x_p2, x_inter], dim=2)
                x_dec = self._cnns_2[i](xi)
                x = self.actf(self._dropout(self._linears_2[i](x_dec)))
                if self.args["extrinsic"] and i != self.args["dec_iterations"] - 1:
                    x = x - x_inter

                prior = self.deinterleaver(x)
        else:
            for i in range(self.args["dec_iterations"]):
                xi = torch.cat([x_sys, x_p1_deint, prior], dim=2)
                x_dec = self._cnns_1[i](xi)
                x = self.actf(self._dropout(self._linears_1[i](x_dec)))
                if self.args["extrinsic"]:
                    x = x - prior

                x_inter = self.interleaver(x)
                xi = torch.cat([x_sys_inter, x_p1, x_inter], dim=2)
                x_dec = self._cnns_2[i](xi)
                x = self.actf(self._dropout(self._linears_2[i](x_dec)))
                if self.args["extrinsic"] and i != self.args["dec_iterations"] - 1:
                    x = x - x_inter

                prior = self.deinterleaver(x)
        if self.args["batch_norm"]:
            x = self._batch_norm_1(prior)
        x = torch.sigmoid(prior)
        return x

class DecoderCNN_nolat(DecoderBase):
    def __init__(self, arguments):
        """
        CNN based decoder with an de/interleaver.
        :param arguments: Arguments as dictionary.
        """
        super(DecoderCNN_nolat, self).__init__(arguments)

        self.interleaver = Interleaver()
        self.deinterleaver = DeInterleaver()

        self._dropout = torch.nn.Dropout(self.args["dec_dropout"])

        self._cnns_1 = torch.nn.ModuleList()
        self._cnns_2 = torch.nn.ModuleList()
        self._linears_1 = torch.nn.ModuleList()
        self._linears_2 = torch.nn.ModuleList()

        if self.args["batch_norm"]:
            self._batch_norm_1 = torch.nn.BatchNorm1d(self.args["block_length"])

        for i in range(self.args["dec_iterations"]):
            self._cnns_1.append(Conv1d(self.args["dec_actf"],
                                       layers=self.args["dec_layers"],
                                       in_channels=2 + self.args["dec_inputs"],
                                       out_channels=self.args["dec_units"],
                                       kernel_size=self.args["dec_kernel"]))
            self._linears_1.append(torch.nn.Linear(self.args["dec_units"], self.args["dec_inputs"]))
            self._cnns_2.append(Conv1d(self.args["dec_actf"],
                                       layers=self.args["dec_layers"],
                                       in_channels=2 + self.args["dec_inputs"],
                                       out_channels=self.args["dec_units"],
                                       kernel_size=self.args["dec_kernel"]))
            if i == self.args["dec_iterations"] - 1:
                self._linears_2.append(torch.nn.Linear(self.args["dec_units"], 1))
            else:
                self._linears_2.append(torch.nn.Linear(self.args["dec_units"], self.args["dec_inputs"]))

    def set_interleaver_order(self, array):
        """
        Inheritance function to set the models interleaver/de-interleaver order.
        :param array: That array that is needed to set/restore interleaver order.
        """
        self.interleaver.set_order(array)
        self.deinterleaver.set_order(array)

    def set_parallel(self):
        """
        Ensures that forward and backward propagation operations can be performed on multiple GPUs.
        """
        self.is_parallel = True
        if self.args["batch_norm"]:
            self._batch_norm_1 = torch.nn.DataParallel(self._batch_norm_1)
        for i in range(self.args["dec_iterations"]):
            self._cnns_1[i] = torch.nn.DataParallel(self._cnns_1[i])
            self._cnns_2[i] = torch.nn.DataParallel(self._cnns_2[i])
            self._linears_1[i] = torch.nn.DataParallel(self._linears_1[i])
            self._linears_2[i] = torch.nn.DataParallel(self._linears_2[i])

    def forward(self, inputs):
        """
        Calculates output tensors from input tensors based on the process.
        :param inputs: Input tensor.
        :return: Output tensor of decoder.
        """
        x_sys = inputs[:, :, 0].view((inputs.size()[0], inputs.size()[1], 1))
        x_sys_inter = self.interleaver(x_sys)
        x_p1 = inputs[:, :, 1].view((inputs.size()[0], inputs.size()[1], 1))
        if self.args["rate"] == "onethird":
            x_p2 = inputs[:, :, 2].view((inputs.size()[0], inputs.size()[1], 1))
        else:
            x_p1_deint = self.deinterleaver(x_p1)

        prior = torch.zeros((inputs.size()[0], inputs.size()[1], self.args["dec_inputs"]))

        if self.args["rate"] == "onethird":
            for i in range(self.args["dec_iterations"]):
                xi = torch.cat([x_sys, x_p1, prior], dim=2)
                x_dec = self._cnns_1[i](xi)
                x = self.actf(self._dropout(self._linears_1[i](x_dec)))
                if self.args["extrinsic"]:
                    x = x - prior

                x_inter = self.interleaver(x)
                xi = torch.cat([x_sys_inter, x_p2, x_inter], dim=2)
                x_dec = self._cnns_2[i](xi)
                x = self.actf(self._dropout(self._linears_2[i](x_dec)))
                if self.args["extrinsic"] and i != self.args["dec_iterations"] - 1:
                    x = x - x_inter

                prior = self.deinterleaver(x)
        else:
            for i in range(self.args["dec_iterations"]):
                xi = torch.cat([x_sys, x_p1_deint, prior], dim=2)
                x_dec = self._cnns_1[i](xi)
                x = self.actf(self._dropout(self._linears_1[i](x_dec)))
                if self.args["extrinsic"]:
                    x = x - prior

                x_inter = self.interleaver(x)
                xi = torch.cat([x_sys_inter, x_p1, x_inter], dim=2)
                x_dec = self._cnns_2[i](xi)
                x = self.actf(self._dropout(self._linears_2[i](x_dec)))
                if self.args["extrinsic"] and i != self.args["dec_iterations"] - 1:
                    x = x - x_inter

                prior = self.deinterleaver(x)
        if self.args["batch_norm"]:
            x = self._batch_norm_1(prior)
        x = torch.sigmoid(prior)
        return x

class DecoderRNNatt(DecoderBase):
    def __init__(self, arguments):
        """
        RNN based decoder with a code rate of 1/3 and an de/interleaver.

        :param arguments: Arguments as dictionary.
        """
        super(DecoderRNNatt, self).__init__(arguments)
        self.interleaver = Interleaver()
        self.deinterleaver = DeInterleaver()
        self.attn = torch.nn.Linear(self.args["dec_units"] * 2, self.args["batch_size"])  # self.args["block_length"])
        self.attn_combine = torch.nn.Linear(self.args["dec_units"] * 2, self.args["dec_units"])

        rnn = torch.nn.RNN
        if self.args["dec_rnn"].lower() == 'gru':
            rnn = torch.nn.GRU
        elif self.args["dec_rnn"].lower() == 'lstm':
            rnn = torch.nn.LSTM

        self._dropout = torch.nn.Dropout(self.args["dec_dropout"])

        self._rnns_1 = torch.nn.ModuleList()
        self._rnns_2 = torch.nn.ModuleList()
        self._linears_1 = torch.nn.ModuleList()
        self._linears_2 = torch.nn.ModuleList()

        for i in range(self.args["dec_iterations"]):
            self._rnns_1.append(rnn(2 + self.args["dec_inputs"], self.args["dec_units"],
                                    num_layers=self.args["dec_layers"],
                                    bias=True,
                                    batch_first=True,
                                    dropout=0,
                                    bidirectional=True))
            self._linears_1.append(torch.nn.Linear(2 * self.args["dec_units"], self.args["dec_inputs"]))
            self._rnns_2.append(rnn(2 + self.args["dec_inputs"], self.args["dec_units"],
                                    num_layers=self.args["dec_layers"],
                                    bias=True,
                                    batch_first=True,
                                    dropout=0,
                                    bidirectional=True))
            if i == self.args["dec_iterations"] - 1:
                self._linears_2.append(torch.nn.Linear(2 * self.args["dec_units"], 1))
            else:
                self._linears_2.append(torch.nn.Linear(2 * self.args["dec_units"], self.args["dec_inputs"]))

    def set_interleaver_order(self, array):
        """
        Inheritance function to set the models interleaver/de-interleaver order.

        :param array: That array that is needed to set/restore interleaver order.
        """
        self.interleaver.set_order(array)
        self.deinterleaver.set_order(array)

    def set_parallel(self):
        """
        Ensures that forward and backward propagation operations can be performed on multiple GPUs.
        """
        self.is_parallel = True
        for i in range(self.args["dec_iterations"]):
            self._rnns_1[i] = torch.nn.DataParallel(self._rnns_1[i])
            self._rnns_2[i] = torch.nn.DataParallel(self._rnns_2[i])
            self._linears_1[i] = torch.nn.DataParallel(self._linears_1[i])
            self._linears_2[i] = torch.nn.DataParallel(self._linears_2[i])

    def forward(self, inputs, hidden, encoder_outputs):
        """
        Calculates output tensors from input tensors based on the process.

        :param inputs: Input tensor.
        :return: Output tensor of decoder.
        """
        x_sys = inputs[:, :, 0].view((inputs.size()[0], inputs.size()[1], 1))
        x_sys_inter = self.interleaver(x_sys)
        x_p1 = inputs[:, :, 1].view((inputs.size()[0], inputs.size()[1], 1))
        if self.args["rate"] == "onethird":
            x_p2 = inputs[:, :, 2].view((inputs.size()[0], inputs.size()[1], 1))

        enc_out = encoder_outputs[:, :, 0]

        prior = torch.zeros((inputs.size()[0], inputs.size()[1], self.args["dec_inputs"]))

        del inputs

        if self.args["rate"] == "onethird":
            for i in range(self.args["dec_iterations"]):
                # Change batch first=true to false, that should circumvent all the transformations.

                p1_new = x_sys.view((x_p1.size()[0], x_p1.size()[1]))
                cat_tens = torch.cat((p1_new, hidden[1][0]), 1)
                attn_weights_0 = torch.nn.functional.softmax(
                    self.attn(cat_tens), dim=1)
                x_p1 = torch.bmm(attn_weights_0.unsqueeze(0), enc_out.unsqueeze(0))
                x_p1 = x_p1.view((x_p1.size()[1], x_p1.size()[2], 1))
                del cat_tens

                p2_new = x_sys.view((x_p2.size()[0], x_p2.size()[1]))
                cat_tens = torch.cat((p2_new, hidden[1][0]), 1)
                attn_weights_1 = torch.nn.functional.softmax(
                    self.attn(cat_tens), dim=1)
                x_p2 = torch.bmm(attn_weights_1.unsqueeze(0), enc_out.unsqueeze(0))
                x_p2 = x_p1.view((x_p2.size()[1], x_p2.size()[2], 1))
                del cat_tens

                xi = torch.cat([x_sys, x_p1, prior], dim=2)

                if self.is_parallel:
                    self._rnns_1[i].module.flatten_parameters()
                x_dec, hidden[0] = self._rnns_1[i](xi, hidden[0])
                x = self.actf(self._dropout(self._linears_1[i](x_dec)))
                if self.args["extrinsic"]:
                    x = x - prior

                x_inter = self.interleaver(x)
                xi = torch.cat([x_sys_inter, x_p2, x_inter], dim=2)
                if self.is_parallel:
                    self._rnns_2[i].module.flatten_parameters()
                x_dec, hidden[1] = self._rnns_2[i](xi, hidden[1])
                x = self.actf(self._dropout(self._linears_2[i](x_dec)))
                if self.args["extrinsic"] and i != self.args["dec_iterations"] - 1:
                    x = x - x_inter

                prior = self.deinterleaver(x)
        else:
            for i in range(self.args["dec_iterations"]):
                # Change batch first=true to false, that should circumvent all the transformations.

                p1_new = x_sys.view((x_p1.size()[0], x_p1.size()[1]))
                cat_tens = torch.cat((p1_new, hidden[1][0]), 1)
                attn_weights_0 = torch.nn.functional.softmax(
                    self.attn(cat_tens), dim=1)
                x_p1 = torch.bmm(attn_weights_0.unsqueeze(0), enc_out.unsqueeze(0))
                x_p1 = x_p1.view((x_p1.size()[1], x_p1.size()[2], 1))
                x_p1_deint = self.deinterleaver(x_p1)
                del cat_tens

                xi = torch.cat([x_sys, x_p1_deint, prior], dim=2)

                if self.is_parallel:
                    self._rnns_1[i].module.flatten_parameters()
                x_dec, hidden[0] = self._rnns_1[i](xi, hidden[0])
                x = self.actf(self._dropout(self._linears_1[i](x_dec)))
                if self.args["extrinsic"]:
                    x = x - prior

                x_inter = self.interleaver(x)
                xi = torch.cat([x_sys_inter, x_p1, x_inter], dim=2)
                if self.is_parallel:
                    self._rnns_2[i].module.flatten_parameters()
                x_dec, hidden[1] = self._rnns_2[i](xi, hidden[1])
                x = self.actf(self._dropout(self._linears_2[i](x_dec)))
                if self.args["extrinsic"] and i != self.args["dec_iterations"] - 1:
                    x = x - x_inter

                prior = self.deinterleaver(x)
        x = torch.sigmoid(prior)
        return x, hidden, [attn_weights_0, attn_weights_1]

    def initHidden(self):
        return 2 * [torch.zeros(2 * self.args["enc_layers"], self.args["batch_size"], self.args[
            "enc_units"])]  # [torch.zeros(10, 256, self.args["enc_units"]), torch.zeros(10, 256, self.args["enc_units"]), torch.zeros(10, 256, self.args["enc_units"])]


class DecoderEnTransformer(DecoderBase):
    def __init__(self, arguments):
        """
        Transformer based decoder with an de/interleaver.

        :param arguments: Arguments as dictionary.
        """
        super(DecoderEnTransformer, self).__init__(arguments)

        self.interleaver = Interleaver()
        self.deinterleaver = DeInterleaver()

        self._dropout = torch.nn.Dropout(self.args["dec_dropout"])

        self._transformers_1 = torch.nn.ModuleList()
        self._transformers_2 = torch.nn.ModuleList()
        self._linears_1 = torch.nn.ModuleList()
        self._linears_2 = torch.nn.ModuleList()

        for i in range(self.args["dec_iterations"]):
            decoder_layer_1 = torch.nn.TransformerEncoderLayer(d_model=self.args["dec_units"],
                                                                     nhead=self.args["dec_kernel"],
                                                                     dropout=self.args["dec_dropout"],
                                                                     activation='relu',
                                                                     # only relu or gelu work as activation function
                                                                     batch_first=True)
            self._transformers_1.append(torch.nn.TransformerEncoder(decoder_layer_1, num_layers=self.args["dec_layers"]))
            self._linears_1.append(torch.nn.Linear(self.args["dec_units"], self.args["dec_inputs"]))
            decoder_layer_2 = torch.nn.TransformerEncoderLayer(d_model=self.args["dec_units"],
                                                                     nhead=self.args["dec_kernel"],
                                                                     dropout=self.args["dec_dropout"],
                                                                     activation='relu',
                                                                     # only relu or gelu work as activation function
                                                                     batch_first=True)
            self._transformers_2.append(torch.nn.TransformerEncoder(decoder_layer_2, num_layers=self.args["dec_layers"]))
            if i == self.args["dec_iterations"] - 1:
                self._linears_2.append(torch.nn.Linear(self.args["dec_units"], 1))
            else:
                self._linears_2.append(torch.nn.Linear(self.args["dec_units"], self.args["dec_inputs"])) #self.args["dec_inputs"]


    def set_interleaver_order(self, array):
        """
        Inheritance function to set the models interleaver/de-interleaver order.

        :param array: That array that is needed to set/restore interleaver order.
        """
        self.interleaver.set_order(array)
        self.deinterleaver.set_order(array)

    def set_parallel(self):
        """
        Ensures that forward and backward propagation operations can be performed on multiple GPUs.
        """
        self.is_parallel = True
        for i in range(self.args["dec_iterations"]):
            self._transformers_1[i] = torch.nn.DataParallel(self._transformers_1[i])
            self._transformers_2[i] = torch.nn.DataParallel(self._transformers_2[i])
            self._linears_1[i] = torch.nn.DataParallel(self._linears_1[i])
            self._linears_2[i] = torch.nn.DataParallel(self._linears_2[i])

    def forward(self, inputs):
        """
        Calculates output tensors from input tensors based on the process.

        :param inputs: Input tensor.
        :return: Output tensor of decoder.
        """
        #ToDo the x_train inputs might have to be interleaved
        #inputs = inputs.permute(0, 2, 1)
        #x_train = x_train.permute(0, 2, 1)

        x_sys = inputs[:, :, 0].view((inputs.size()[0], inputs.size()[1], 1))
        x_sys_inter = self.interleaver(x_sys)
        x_p1 = inputs[:, :, 1].view((inputs.size()[0], inputs.size()[1], 1))
        if self.args["rate"] == "onethird":
            x_p2 = inputs[:, :, 2].view((inputs.size()[0], inputs.size()[1], 1))
        else:
            x_p1_deint = self.deinterleaver(x_p1)
        prior = torch.zeros((inputs.size()[0], inputs.size()[1], self.args["dec_inputs"])) #self.args["dec_inputs"]
        if self.args["rate"] == "onethird":
            for i in range(self.args["dec_iterations"]):
                xi = torch.cat([x_sys, x_p1, prior], dim=2)
                if self.is_parallel:
                    self._transformers_1[i].module.flatten_parameters()

                x_dec = self._transformers_1[i](xi) #x_train,
                x = self.actf(self._dropout(self._linears_1[i](x_dec)))
                if self.args["extrinsic"]:
                    x = x - prior
                x_inter = self.interleaver(x)
                xi = torch.cat([x_sys_inter, x_p2, x_inter], dim=2)
                if self.is_parallel:
                    self._transformers_2[i].module.flatten_parameters()
                x_dec = self._transformers_2[i](xi) #x_train,
                x = self.actf(self._dropout(self._linears_2[i](x_dec)))
                if self.args["extrinsic"] and i != self.args["dec_iterations"] - 1:
                    x = x - x_inter

                prior = self.deinterleaver(x)
        else:
            for i in range(self.args["dec_iterations"]):
                xi = torch.cat([x_sys, x_p1_deint, prior], dim=2)
                if self.is_parallel:
                    self._transformers_1[i].module.flatten_parameters()
                x_dec = self._transformers_1[i](xi) #x_train,
                x = self.actf(self._dropout(self._linears_1[i](x_dec)))
                if self.args["extrinsic"]:
                    x = x - prior

                x_inter = self.interleaver(x)
                xi = torch.cat([x_sys_inter, x_p1, x_inter], dim=2)
                if self.is_parallel:
                    self._transformers_2[i].module.flatten_parameters()
                x_dec = self._transformers_2[i](xi) #x_train,
                x = self.actf(self._dropout(self._linears_2[i](x_dec)))
                if self.args["extrinsic"] and i != self.args["dec_iterations"] - 1:
                    x = x - x_inter

                prior = self.deinterleaver(x)

        x = torch.sigmoid(prior)
        return x
class DecoderTransformer(DecoderBase):
    def __init__(self, arguments):
        """
        Transformer based decoder with an de/interleaver.

        :param arguments: Arguments as dictionary.
        """
        super(DecoderTransformer, self).__init__(arguments)

        self.interleaver = Interleaver()
        self.deinterleaver = DeInterleaver()

        self._dropout = torch.nn.Dropout(self.args["dec_dropout"])

        self._transformers_1 = torch.nn.ModuleList()
        self._transformers_2 = torch.nn.ModuleList()
        self._linears_1 = torch.nn.ModuleList()
        self._linears_2 = torch.nn.ModuleList()

        #self.args["dec_units"]
        for i in range(self.args["dec_iterations"]):
            #decoder_layer_11 = torch.nn.TransformerDecoderLayer(d_model=self.args["dec_units"],
            #                                                  nhead=self.args["dec_kernel"],
            #                                                  dropout=self.args["enc_dropout"],
            #                                                  activation='relu',
                                                              # only relu or gelu work as activation function
            #                                                  batch_first=True)
            decoder_layer_1 = torch.nn.TransformerDecoderLayer(d_model=self.args["dec_units"],
                                                                     nhead=self.args["dec_kernel"],
                                                                     dropout=self.args["dec_dropout"],
                                                                     activation='relu',
                                                                     # only relu or gelu work as activation function
                                                                     batch_first=True)
            self._transformers_1.append(torch.nn.TransformerDecoder(decoder_layer_1, num_layers=self.args["dec_layers"]))
            self._linears_1.append(torch.nn.Linear(self.args["dec_units"], self.args["dec_inputs"])) #self.args["dec_inputs"]
            decoder_layer_2 = torch.nn.TransformerDecoderLayer(d_model=self.args["dec_units"],
                                                                     nhead=self.args["dec_kernel"],
                                                                     dropout=self.args["dec_dropout"],
                                                                     activation='relu',
                                                                     # only relu or gelu work as activation function
                                                                     batch_first=True)
           # decoder_layer_2 = torch.nn.TransformerDecoderLayer(d_model=self.args["dec_units"],
           #                                                   nhead=self.args["dec_kernel"],
           #                                                   dropout=self.args["enc_dropout"],
           #                                                   activation='relu',
           #                                                   # only relu or gelu work as activation function
           #                                                   batch_first=True)
            self._transformers_2.append(torch.nn.TransformerDecoder(decoder_layer_2, num_layers=self.args["dec_layers"]))
            if i == self.args["dec_iterations"] - 1:
                self._linears_2.append(torch.nn.Linear(self.args["dec_units"], 1))
            else:
                self._linears_2.append(torch.nn.Linear(self.args["dec_units"], self.args["dec_inputs"])) #self.args["dec_inputs"]


    def set_interleaver_order(self, array):
        """
        Inheritance function to set the models interleaver/de-interleaver order.

        :param array: That array that is needed to set/restore interleaver order.
        """
        self.interleaver.set_order(array)
        self.deinterleaver.set_order(array)

    def set_parallel(self):
        """
        Ensures that forward and backward propagation operations can be performed on multiple GPUs.
        """
        self.is_parallel = True
        for i in range(self.args["dec_iterations"]):
            self._transformers_1[i] = torch.nn.DataParallel(self._transformers_1[i])
            self._transformers_2[i] = torch.nn.DataParallel(self._transformers_2[i])
            self._linears_1[i] = torch.nn.DataParallel(self._linears_1[i])
            self._linears_2[i] = torch.nn.DataParallel(self._linears_2[i])

    def forward(self, inputs, x_train):
        """
        Calculates output tensors from input tensors based on the process.

        :param inputs: Input tensor.
        :return: Output tensor of decoder.
        """
        #ToDo the x_train inputs might have to be interleaved
        #inputs = inputs.permute(0, 2, 1)
        #x_train = x_train.permute(0, 2, 1)

        x_sys = inputs[:, :, 0].view((inputs.size()[0], inputs.size()[1], 1))
        x_train_sys = x_train[:, :, 0].view((x_train.size()[0], x_train.size()[1], 1))
        x_sys_inter = self.interleaver(x_sys)
        x_train_inter = self.interleaver(x_train_sys)
        x_p1 = inputs[:, :, 1].view((inputs.size()[0], inputs.size()[1], 1))
        x_train_p1 = x_train[:, :, 1].view((x_train.size()[0], x_train.size()[1], 1))
        if self.args["rate"] == "onethird":
            x_p2 = inputs[:, :, 2].view((inputs.size()[0], inputs.size()[1], 1))
            x_train_p2 = x_train[:, :, 2].view((x_train.size()[0], x_train.size()[1], 1))
        else:
            x_p1_deint = self.deinterleaver(x_p1)
            x_train_p1_deint = self.deinterleaver(x_train_p1)
        prior = torch.zeros((inputs.size()[0], inputs.size()[1], self.args["dec_inputs"])) #self.args["dec_inputs"]
        if self.args["rate"] == "onethird":
            for i in range(self.args["dec_iterations"]):
                xi = torch.cat([x_sys, x_p1, prior], dim=2)
                x_train_i = torch.cat([x_train_sys, x_train_p1, prior], dim=2) #TBH it makes little sense to put the prior (and further down, x_inter) into the encoder output part of the transformer
                if self.is_parallel:
                    self._transformers_1[i].module.flatten_parameters()

                x_dec = self._transformers_1[i](x_train_i, xi) #x_train,
                x = self.actf(self._dropout(self._linears_1[i](x_dec)))
                if self.args["extrinsic"]:
                    x = x - prior
                x_inter = self.interleaver(x)
                xi = torch.cat([x_sys_inter, x_p2, x_inter], dim=2)
                x_train_i = torch.cat([x_train_inter, x_train_p2, x_inter], dim=2)
                if self.is_parallel:
                    self._transformers_2[i].module.flatten_parameters()
                x_dec = self._transformers_2[i](x_train_i, xi) #x_train,
                x = self.actf(self._dropout(self._linears_2[i](x_dec)))
                if self.args["extrinsic"] and i != self.args["dec_iterations"] - 1:
                    x = x - x_inter

                prior = self.deinterleaver(x)
        else:
            for i in range(self.args["dec_iterations"]):
                xi = torch.cat([x_sys, x_p1_deint, prior], dim=2)
                x_train_i = torch.cat([x_train_sys, x_train_p1_deint, prior], dim=2)
                if self.is_parallel:
                    self._transformers_1[i].module.flatten_parameters()
                x_dec = self._transformers_1[i](x_train_i, xi) #x_train,
                x = self.actf(self._dropout(self._linears_1[i](x_dec)))
                if self.args["extrinsic"]:
                    x = x - prior

                x_inter = self.interleaver(x)
                xi = torch.cat([x_sys_inter, x_p1, x_inter], dim=2)
                x_train_i = torch.cat([x_train_inter, x_train_p1, x_inter], dim=2)
                if self.is_parallel:
                    self._transformers_2[i].module.flatten_parameters()
                x_dec = self._transformers_2[i](x_train_i, xi) #x_train,
                x = self.actf(self._dropout(self._linears_2[i](x_dec)))
                if self.args["extrinsic"] and i != self.args["dec_iterations"] - 1:
                    x = x - x_inter

                prior = self.deinterleaver(x)

        x = torch.sigmoid(prior)
        return x


class EnsembleDecoder(DecoderBase):
    def __init__(self, arguments):
        """
        Ensemble of CNN based networks. Used to reconstruct the code stream from the encoder after
        applying the noisy channel.
        :param arguments: Arguments as dictionary.
        """
        super(EnsembleDecoder, self).__init__(arguments)

        self.n_models = self.args['n_models']
        self.models = torch.nn.ModuleList([DecoderCNN_nolat(self.args) for i in range(self.n_models)])

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

class DecoderResNet1d(DecoderBase):
    def __init__(self, arguments):
        """
        CNN based decoder with an de/interleaver.
        :param arguments: Arguments as dictionary.
        """
        super(DecoderResNet1d, self).__init__(arguments)

        self.interleaver = Interleaver()
        self.deinterleaver = DeInterleaver()

        self._dropout = torch.nn.Dropout(self.args["dec_dropout"])

        self._cnns_1 = torch.nn.ModuleList()
        self._cnns_2 = torch.nn.ModuleList()
        self._linears_1 = torch.nn.ModuleList()
        self._linears_2 = torch.nn.ModuleList()
        self._layers_1 = list()
        self._layers_2 = list()

        if self.args["batch_norm"]:
            self._batch_norm_1 = torch.nn.BatchNorm1d(self.args["block_length"])

        for i in range(self.args["dec_iterations"]):
            self._cnns_1.append(Conv1d(self.args["dec_actf"],
                                       layers=self.args["dec_layers"],
                                       in_channels=2 + self.args["dec_inputs"],
                                       out_channels=self.args["dec_units"],
                                       kernel_size=self.args["dec_kernel"]))
            self._linears_1.append(torch.nn.Linear(self.args["dec_units"], self.args["dec_inputs"]))
            self._cnns_2.append(Conv1d(self.args["dec_actf"],
                                       layers=self.args["dec_layers"],
                                       in_channels=2 + self.args["dec_inputs"],
                                       out_channels=self.args["dec_units"],
                                       kernel_size=self.args["dec_kernel"]))
            layers_1 = []
            for j in range(5):
                layers_1.append(ResNetBlock(self.args["enc_actf"],
                             layers=self.args["dec_layers"],
                             in_channels=self.args["dec_units"],
                             out_channels=self.args["dec_units"],
                             kernel_size=self.args["dec_kernel"],
                            block_length=self.args["block_length"]))
            self._layers_1.append(torch.nn.Sequential(*layers_1))

            layers_2 = []
            for k in range(5):
                layers_2.append(ResNetBlock(self.args["enc_actf"],
                             layers=self.args["dec_layers"],
                             in_channels=self.args["dec_units"],
                             out_channels=self.args["dec_units"],
                             kernel_size=self.args["dec_kernel"],
                            block_length=self.args["block_length"]))
            self._layers_2.append(torch.nn.Sequential(*layers_2))

            if i == self.args["dec_iterations"] - 1:
                self._linears_2.append(torch.nn.Linear(self.args["dec_units"], 1))
            else:
                self._linears_2.append(torch.nn.Linear(self.args["dec_units"], self.args["dec_inputs"]))

    def set_interleaver_order(self, array):
        """
        Inheritance function to set the models interleaver/de-interleaver order.
        :param array: That array that is needed to set/restore interleaver order.
        """
        self.interleaver.set_order(array)
        self.deinterleaver.set_order(array)

    def set_parallel(self):
        """
        Ensures that forward and backward propagation operations can be performed on multiple GPUs.
        """
        self.is_parallel = True
        if self.args["batch_norm"]:
            self._batch_norm_1 = torch.nn.DataParallel(self._batch_norm_1)
        for i in range(self.args["dec_iterations"]):
            self._cnns_1[i] = torch.nn.DataParallel(self._cnns_1[i])
            self._cnns_2[i] = torch.nn.DataParallel(self._cnns_2[i])
            self._linears_1[i] = torch.nn.DataParallel(self._linears_1[i])
            self._linears_2[i] = torch.nn.DataParallel(self._linears_2[i])

    def forward(self, inputs):
        """
        Calculates output tensors from input tensors based on the process.
        :param inputs: Input tensor.
        :return: Output tensor of decoder.
        """
        x_sys = inputs[:, :, 0].view((inputs.size()[0], inputs.size()[1], 1))
        x_sys_inter = self.interleaver(x_sys)
        x_p1 = inputs[:, :, 1].view((inputs.size()[0], inputs.size()[1], 1))
        if self.args["rate"] == "onethird":
            x_p2 = inputs[:, :, 2].view((inputs.size()[0], inputs.size()[1], 1))
        else:
            x_p1_deint = self.deinterleaver(x_p1)

        prior = torch.zeros((inputs.size()[0], inputs.size()[1], self.args["dec_inputs"]))

        if self.args["rate"] == "onethird":
            for i in range(self.args["dec_iterations"]):
                xi = torch.cat([x_sys, x_p1, prior], dim=2)
                x_dec = self._cnns_1[i](xi)
                x_dec = self._layers_1[i](x_dec)
                x = self.actf(self._dropout(self._linears_1[i](x_dec)))
                if self.args["extrinsic"]:
                    x = x - prior

                x_inter = self.interleaver(x)
                xi = torch.cat([x_sys_inter, x_p2, x_inter], dim=2)
                x_dec = self._cnns_2[i](xi)
                x_dec = self._layers_2[i](x_dec)
                x = self.actf(self._dropout(self._linears_2[i](x_dec)))
                if self.args["extrinsic"] and i != self.args["dec_iterations"] - 1:
                    x = x - x_inter

                prior = self.deinterleaver(x)
        else:
            for i in range(self.args["dec_iterations"]):
                xi = torch.cat([x_sys, x_p1_deint, prior], dim=2)
                x_dec = self._cnns_1[i](xi)
                x_dec = self._layers_1[i](x_dec)
                x = self.actf(self._dropout(self._linears_1[i](x_dec)))
                if self.args["extrinsic"]:
                    x = x - prior

                x_inter = self.interleaver(x)
                xi = torch.cat([x_sys_inter, x_p1, x_inter], dim=2)
                x_dec = self._cnns_2[i](xi)
                x_dec = self._layers_2[i](x_dec)
                x = self.actf(self._dropout(self._linears_2[i](x_dec)))
                if self.args["extrinsic"] and i != self.args["dec_iterations"] - 1:
                    x = x - x_inter

                prior = self.deinterleaver(x)
        if self.args["batch_norm"]:
            x = self._batch_norm_1(prior)
        x = torch.sigmoid(prior)
        return x

class ResNetBlock(torch.nn.Module):
    def __init__(self, actf, layers, in_channels, out_channels, kernel_size,block_length):
        super(ResNetBlock, self).__init__()

        self.conv1 = Conv1d(actf, layers, in_channels, out_channels, kernel_size)
        self.bn1 = torch.nn.BatchNorm1d(block_length)
        self.conv2 = Conv1d(actf, layers, in_channels, out_channels, kernel_size)
        self.bn2 = torch.nn.BatchNorm1d(block_length)

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