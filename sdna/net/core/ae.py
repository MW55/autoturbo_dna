# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as func
import numpy as np


class AutoEncoder(torch.nn.Module):
    def __init__(self, arguments, enc, dec, coder, channel):
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

    def forward(self, inputs, padding=0, seed=0, hidden=None):
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

        if self.args["encoder"] == "rnnatt":
            x, hidden = self.enc(inputs, hidden)
        else:
            x = self.enc(inputs)        # stream encoder => in (0, +1) | out (-1, +1)
        s_enc = x.clone()
        noise = self.channel.generate_noise(x, padding, seed)
        noise = noise.cuda() if self.args["gpu"] else noise

        if padding <= 0:
            x *= noise                # noisy channel => in (-1, +1) | out (-1, +1)
            c_dec = []
            if self.args["decoder"] == "rnnatt":
                s_dec = self.dec(x, hidden, s_enc)
            elif self.args["decoder"] == "transformer":
                s_dec = self.dec(x, s_enc)
            else:
                s_dec = self.dec(x)       # stream decoder => in (-1, +1) | out (0, +1)
        else:
            x = func.pad(input=x, pad=(0, 0, 0, padding), mode="constant", value=1.0)
            x *= noise                  # noisy channel => in (-1, +1) | out (-1, 0, +1)
            #x += noise                # some noise must be additive applied (only for testing)
            c_dec = self.coder(x)       # channel decoder => in (-1, 0, +1) | out (-1, +1)
            if self.args["decoder"] == "rnnatt":
                s_dec = self.dec(c_dec, hidden, s_enc)
            elif self.args["decoder"] == "transformer":
                s_dec = self.dec(c_dec, inputs)
            else:
                s_dec = self.dec(c_dec)  # stream decoder => in (-1, +1) | out (0, +1)
        return s_dec, s_enc, c_dec, x
