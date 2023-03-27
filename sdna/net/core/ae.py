# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as func
import numpy as np
#from sdna.net.functional.train import get_same_packages


class AutoEncoder(torch.nn.Module):
    def __init__(self, arguments, enc, dec, coder, channel, coder2=None, coder3=None):
        """
        Autoencoder combines the encoder, decoder as well as the coder and applies the 'noisy' channel. The combination of
        all three networks maps the model.

        :param arguments: Arguments as dictionary.
        :param enc: Model of the encoder.
        :param dec: Model of the decoder.
        :param coder: Model of the coder.
        :param channel: Prepared channel class.
        """
        super(AutoEncoder, self).__init__()
        self.args = arguments
        self.enc = enc
        self.dec = dec
        self.coder = coder
        self.channel = channel
        self.coder2 = coder2
        self.coder3 = coder3

    def forward(self, inputs, padding=0, seed=0, hidden=None, validate=False):
        """
        Calculates output tensors from input tensors based on the process.

        :param inputs: Input tensors.
        :param padding: The number of bits by which the output of the channel should be extended.
        :param seed: Specify a integer number, this allows to reproduce the results.
        :return: Tuple consisting of output tensors of decoder, encoder, coder and channel.

        :note: If the padding is set to zero, the coder is ignored.
        """
        # setup/update interleaver
        ao_enc = np.random.mtrand.RandomState(seed).permutation(np.arange(inputs.size()[1]))
        self.enc.set_interleaver_order(ao_enc)
        self.dec.set_interleaver_order(ao_enc)
        if self.args["coder"] == "resnet2d":
            self.coder.set_interleaver_order(ao_enc)
        if self.args["decoder"] == 'ensemble_dec':
            for model in self.dec.models:
                model.set_interleaver_order(ao_enc)

        if self.args["encoder"] == "rnnatt":
            x, hidden = self.enc(inputs, hidden)
        else:
            x = self.enc(inputs)        # stream encoder => in (0, +1) | out (-1, +1)
        s_enc = x.clone()
        if not self.args["continuous"]:
            noise = self.channel.generate_noise(x, padding, seed, validate, self.args["channel"])
            noise = noise.cuda() if self.args["gpu"] else noise

        if padding <= 0:
            if self.args["continuous"] or self.args["channel"] in ('basic_dna', 'conc_dna'):
                x = self.channel.generate_noise(x, 0, seed, validate, self.args["channel"])
            else:
                x *= noise                # noisy channel => in (-1, +1) | out (-1, +1)
            c_dec = []
            if self.args["decoder"] == "rnnatt":
                s_dec = self.dec(x, hidden, s_enc)
            elif self.args["decoder"] == "transformer":
                #bin_mask = torch.ones((s_enc.size()[0], s_enc.size()[1], s_enc.size()[2]), dtype=torch.bool)
                s_dec = self.dec(x, s_enc)#s_enc
            else:
                s_dec = self.dec(x)       # stream decoder => in (-1, +1) | out (0, +1)
        else:
            x = pad_data(x, padding)
            if self.args["continuous"] or self.args["channel"] in ("basic_dna", "conc_dna"):
                x = self.channel.generate_noise(x, padding, seed, validate, self.args["channel"])
            else:
                x *= noise                  # noisy channel => in (-1, +1) | out (-1, 0, +1)

            #x += noise                # some noise must be additive applied (only for testing)

            '''
            print("sys:")
            get_same_packages(x[:, :, 0].view((x.size()[0], x.size()[1], 1)),
                              s_enc[:, :, 0].view((s_enc.size()[0], s_enc.size()[1], 1)), 2, 0)
            print("p1:")
            get_same_packages(x[:, :, 1].view((x.size()[0], x.size()[1], 1)),
                              s_enc[:, :, 1].view((s_enc.size()[0], s_enc.size()[1], 1)), 2, 0)
            '''
            if self.args["coder"] == 'idt':
                padded_enc = pad_data(s_enc, padding)
                c_dec = self.coder(x, padded_enc)       # channel decoder => in (-1, 0, +1) | out (-1, +1)
            elif self.coder2 and self.coder3:
                x_sys = x[:, :, 0].view((x.size()[0], x.size()[1], 1))
                x_p1 = x[:, :, 1].view((x.size()[0], x.size()[1], 1))
                x_p2 = x[:, :, 2].view((x.size()[0], x.size()[1], 1))

                #print("sys: " + str(torch.any(x_sys == 0)))
                #print("p1: " + str(torch.any(x_p1 == 0)))
                #print("p2: " + str(torch.any(x_p2 == 0)))
                x_sys_c = self.coder(x_sys)
                x_p1_c = self.coder2(x_p1)
                x_p2_c = self.coder3(x_p2)
                c_dec = torch.cat([x_sys_c, x_p1_c, x_p2_c], dim=2)
            else:
                c_dec = self.coder(x)
            if self.args["decoder"] == "rnnatt":
                s_dec = self.dec(c_dec, hidden, s_enc)
            elif self.args["decoder"] == "transformer":
                if validate:
                    s_dec = self.dec(c_dec, c_dec)
                else:
                    s_dec = self.dec(c_dec, s_enc) #s_enc
            else:
                s_dec = self.dec(c_dec)  # stream decoder => in (-1, +1) | out (0, +1)
        return s_dec, s_enc, c_dec, x

def pad_data(x, padding):
    # x = func.pad(input=x, pad=(0, 0, int(padding/2), int(padding/2)), mode="constant", value=1.0)
    # x = func.pad(x, (0, 0, int(padding / 2), int(padding / 2)), mode='circular')
    # x = func.pad(x, (int(padding/2), int(padding/2)), mode='circular')
    x = func.pad(input=x, pad=(0, 0, 0, padding), mode="constant", value=1.0)
    #x = x.permute(0, 2, 1)
    #x = func.pad(x, (0, padding), mode='circular')
    #x = x.permute(0, 2, 1)
    return x