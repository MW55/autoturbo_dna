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
        #For now trying it after the iterative decoding, it should also be tried inside the iterative decoding or before
        self._batch_norm_1 = torch.nn.BatchNorm1d(self.args["block_length"])
        """
        self._latent_1_1 = Conv1d(self.args["dec_actf"],
                                      layers=1,
                                      in_channels=self.args["block_length"] + int(self.args["redundancy"]),  # + 16
                                      out_channels=self.args["block_length"] + int(self.args["redundancy"]/2),
                                      kernel_size=self.args["dec_kernel"])
        self._latent_1_2 = Conv1d(self.args["dec_actf"],
                                      layers=1,
                                      in_channels=self.args["block_length"] + int(self.args["redundancy"]/2),  # + 16
                                      out_channels=self.args["block_length"],
                                      kernel_size=self.args["dec_kernel"])
        self._latent_2_1 = Conv1d(self.args["dec_actf"],
                                      layers=1,
                                      in_channels=self.args["block_length"] + int(self.args["redundancy"]),  # + 16
                                      out_channels=self.args["block_length"] + int(self.args["redundancy"]/2),
                                      kernel_size=self.args["dec_kernel"])
        self._latent_2_2 = Conv1d(self.args["dec_actf"],
                                      layers=1,
                                      in_channels=self.args["block_length"] + int(self.args["redundancy"]/2),  # + 16
                                      out_channels=self.args["block_length"],
                                      kernel_size=self.args["dec_kernel"])
        """
        if self.args["rate"] == "onethird":
            self._latent_3_1 = torch.nn.Linear(self.args["block_length"] + int(self.args["redundancy"]),  # + 16
                                              self.args["block_length"] + int(self.args["redundancy"]/2))
            self._latent_3_2 = torch.nn.Linear(self.args["block_length"] + int(self.args["redundancy"]/2),  # + 16
                                              self.args["block_length"])
            """
            self._latent_3_1 = Conv1d(self.args["dec_actf"],
                                          layers=1,
                                          in_channels=self.args["block_length"] + int(self.args["redundancy"]),  # + 16
                                          out_channels=self.args["block_length"] + int(self.args["redundancy"]/2),
                                          kernel_size=self.args["dec_kernel"])
            self._latent_3_2 = Conv1d(self.args["dec_actf"],
                                    layers=1,
                                    in_channels=self.args["block_length"] + int(self.args["redundancy"]/2),  # + 16
                                    out_channels=self.args["block_length"],
                                    kernel_size=self.args["dec_kernel"])
            """
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
        if self.args["rate"] == "onethird":
            self._latent_3_1 = torch.nn.DataParallel(self._latent_3_1)
            self._latent_3_2 = torch.nn.DataParallel(self._latent_3_2)
        for i in range(self.args["dec_iterations"]):
            #self._latents_1[i] = torch.nn.DataParallel(self._latents_1[i])
            #self._latents_2[i] = torch.nn.DataParallel(self._latents_2[i])
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

        #test
        x_sys = torch.flatten(x_sys, start_dim=1)
        #x_sys = x_sys.reshape((inputs.size()[0], self.args["block_length"]+int(self.args["redundancy"]), 1))
        x_sys = self.actf(self._dropout(self._latent_1_1(x_sys)))
        x_sys = self.actf(self._dropout(self._latent_1_2(x_sys)))
        x_sys = x_sys.reshape((inputs.size()[0], self.args["block_length"], 1))

        if bool(torch.isnan(x_sys).any()):
            print("failed")
        #test over

        #x_sys = x_sys.permute(0, 2, 1)
        #x_sys = self._latent_1_1(x_sys)
        #x_sys = self.actf(self._dropout(x_sys)) # new, could dropout lead to different sized outputs?
        #x_sys = self._latent_1_2(x_sys)
        #x_sys = self.actf(self._dropout(x_sys))  # new, could dropout lead to different sized outputs?
        #x_sys = x_sys.permute(0, 2, 1)
        #x_sys = self.actf(self._dropout(x_sys))


        x_sys_inter = self.interleaver(x_sys)
        x_p1 = inputs[:, :, 1].view((inputs.size()[0], inputs.size()[1], 1))

        #test
        x_p1 = torch.flatten(x_p1, start_dim=1)
        #x_p1 = x_p1.reshape((inputs.size()[0], self.args["block_length"]+int(self.args["redundancy"]), 1))
        x_p1 = self.actf(self._dropout(self._latent_2_1(x_p1)))
        x_p1 = self.actf(self._dropout(self._latent_2_2(x_p1)))
        x_p1 = x_p1.reshape((inputs.size()[0], self.args["block_length"], 1))
        #test over


        if bool(torch.isnan(x_p1).any()):
            print("failed")

        #x_p1 = x_p1.permute(0, 2, 1)
        #x_p1 = self._latent_2_1(x_p1)
        #x_p1 = self.actf(self._dropout(x_p1))  # new, could dropout lead to different sized outputs?
        #x_p1 = self._latent_2_2(x_p1)
        #x_p1 = self.actf(self._dropout(x_p1))  # new, could dropout lead to different sized outputs?
        #x_p1 = x_p1.permute(0, 2, 1)
        #x_p1 = self.actf(self._dropout(x_p1))

        if self.args["rate"] == "onethird":
            x_p2 = inputs[:, :, 2].view((inputs.size()[0], inputs.size()[1], 1))

            #test
            x_p2 = torch.flatten(x_p2, start_dim=1)
            #x_p2 = x_p2.reshape((inputs.size()[0], self.args["block_length"]+int(self.args["redundancy"]), 1))
            x_p2 = self.actf(self._dropout(self._latent_3_1(x_p2)))
            x_p2 = self.actf(self._dropout(self._latent_3_2(x_p2)))
            x_p2 = x_p2.reshape((inputs.size()[0], self.args["block_length"], 1))
            #test over

            if bool(torch.isnan(x_p2).any()):
                print("failed")
            #x_p2 = x_p2.permute(0, 2, 1)
            #x_p2 = self._latent_3_1(x_p2)
            #x_p2 = self.actf(self._dropout(x_p2))  # new, could dropout lead to different sized outputs?
            #x_p2 = self._latent_3_2(x_p2)
            #x_p2 = self.actf(self._dropout(x_p2))  # new, could dropout lead to different sized outputs?
            #x_p2 = x_p2.permute(0, 2, 1)
            #x_p2 = self.actf(self._dropout(x_p2))
        else:
            x_p1_deint = self.deinterleaver(x_p1) #ToDo: check if that is right



        prior = torch.zeros((inputs.size()[0], inputs.size()[1]-int(self.args["redundancy"]), self.args["dec_inputs"])) #todo: -16 is hardcoded latent rendundancy! Has to be made an argurment

        if self.args["rate"] == "onethird":
            for i in range(self.args["dec_iterations"]):
                xi = torch.cat([x_sys, x_p1, prior], dim=2)
                x_dec = self._cnns_1[i](xi)

                if bool(torch.isnan(x_dec).any()):
                    print("failed")
                x_dec = self._linears_1[i](x_dec)
                if bool(torch.isnan(x_dec).any()):
                    print("failed")
                x = self.actf(self._dropout(x_dec))

                if bool(torch.isnan(x).any()):
                    print("failed")

                if self.args["extrinsic"]:
                    x = x - prior

                if bool(torch.isnan(x).any()):
                    print("failed")

                x_inter = self.interleaver(x)
                xi = torch.cat([x_sys_inter, x_p2, x_inter], dim=2)
                x_dec = self._cnns_2[i](xi)
                x = self.actf(self._dropout(self._linears_2[i](x_dec)))
                if self.args["extrinsic"] and i != self.args["dec_iterations"] - 1:
                    x = x - x_inter

                prior = self.deinterleaver(x)
        else: # Todo: Check if this is actually correct
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
        x = self._batch_norm_1(prior) #new
        x = torch.sigmoid(x)

        if bool(torch.isnan(x).any()):
            print("failed")

        return x

# CNN Decoder with de/interleaver
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
            encoder_layer1 = torch.nn.TransformerDecoderLayer(d_model=self.args["dec_units"],
                                                              nhead=self.args["dec_kernel"],
                                                              dropout=self.args["enc_dropout"],
                                                              activation='relu',
                                                              # only relu or gelu work as activation function
                                                              batch_first=True)
            self._transformers_1.append(torch.nn.TransformerDecoder(encoder_layer1, num_layers=self.args["dec_layers"]))
            self._linears_1.append(torch.nn.Linear(self.args["dec_units"], self.args["dec_inputs"]))

            encoder_layer2 = torch.nn.TransformerDecoderLayer(d_model=self.args["dec_units"],
                                                              nhead=self.args["dec_kernel"],
                                                              dropout=self.args["enc_dropout"],
                                                              activation='relu',
                                                              # only relu or gelu work as activation function
                                                              batch_first=True)
            self._transformers_1.append(torch.nn.TransformerDecoder(encoder_layer2, num_layers=self.args["dec_layers"]))
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
        inputs = inputs.permute(0, 2, 1)
        x_train = x_train.permute(0, 2, 1)

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
                    self._transformers_1[i].module.flatten_parameters()

                x_dec = self._transformers_1[i](x_train, xi)
                x = self.actf(self._dropout(self._linears_1[i](x_dec)))
                if self.args["extrinsic"]:
                    x = x - prior
                x_inter = self.interleaver(x)
                xi = torch.cat([x_sys_inter, x_p2, x_inter], dim=2)
                if self.is_parallel:
                    self._transformers_2[i].module.flatten_parameters()
                x_dec, _ = self._transformers_2[i](x_train, xi)
                x = self.actf(self._dropout(self._linears_2[i](x_dec)))
                if self.args["extrinsic"] and i != self.args["dec_iterations"] - 1:
                    x = x - x_inter

                prior = self.deinterleaver(x)
        else:
            for i in range(self.args["dec_iterations"]):
                xi = torch.cat([x_sys, x_p1_deint, prior], dim=2)
                if self.is_parallel:
                    self._transformers_1[i].module.flatten_parameters()
                x_dec = self._transformers_1[i](x_train, xi)
                x = self.actf(self._dropout(self._linears_1[i](x_dec)))
                if self.args["extrinsic"]:
                    x = x - prior

                x_inter = self.interleaver(x)
                xi = torch.cat([x_sys_inter, x_p1, x_inter], dim=2)
                if self.is_parallel:
                    self._transformers_2[i].module.flatten_parameters()
                x_dec, _ = self._transformers_2[i](x_train, xi)
                x = self.actf(self._dropout(self._linears_2[i](x_dec)))
                if self.args["extrinsic"] and i != self.args["dec_iterations"] - 1:
                    x = x - x_inter

                prior = self.deinterleaver(x)

        x = torch.sigmoid(prior)
        return x

 #       for i in range(self.args["dec_iterations"]):
 #           x = self._transformer_1(x)
 #           x = self._transformer_2(x)
 #           x = self._linears_1[i](x)
 #           x = torch.nn.functional.relu(x)
 #           x = self._dropout(x)
 #           x = self._linears_2[i](x)
 #           x = torch.nn.functional.relu(x)
 #           x = self._dropout(x)
 #       x_sys_deinter = self.deinterleaver(x)
 #       return x_sys_deinter

#    def forward(self, inputs):
#        """
#        #Calculates output tensors from input tensors based on the process.

#:param inputs: Input tensor.
#:return: Output tensor of decoder.
#        """
#        x_sys = inputs[:, :, 0].view((inputs.size()[0], inputs.size()[1], 1))
#        x_sys_inter = self.interleaver(x_sys)
#        x_p1 = inputs[:, :, 1].view((inputs.size()[0], inputs.size()[1], 1))
#        if self.args["rate"] == "onethird":
#            x_p2 = inputs[:, :, 2].view((inputs.size()[0], inputs.size()[1], 1))
#            x = torch.cat((x_sys_inter, x_p1, x_p2), 2)
#        else:
#            x = torch.cat((x_sys_inter, x_p1), 2)

#        for i in range(self.args["dec_iterations"]):
#            x = self._transformer_1(x)
#            x = self._transformer_2(x)
#            x = self._linears_1[i](x)
#            x = torch.nn.functional.relu(x)
#            x = self._dropout(x)
#            x = self._linears_2[i](x)
#            x = torch.nn.functional.relu(x)
#            x = self._dropout(x)
#        x_sys_deinter = self.deinterleaver(x)
#        return x_sys_deinter
