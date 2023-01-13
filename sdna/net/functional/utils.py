# -*- coding: utf-8 -*-

import numpy as np
import torch

from sdna.net.core import *


def encode(args, net, in_arr=None):
    """
    Helper function for encoding.

    :param args: Arguments as dictionary.
    :param net: Instance of the class net.
    :param in_arr: Array of bits to be encoded, if the input is a file.
    :returns: Encoded bit stream.
    """
    # prepare bits
    if args["bitenc"]:
        x = np.array(list(args["bitenc"]), dtype=np.float32)
    else:
        x = in_arr
    x = np.reshape(x, (-1, args["block_length"], 1))
    x = torch.from_numpy(x).cuda() if args["gpu"] else torch.from_numpy(x)

    # try to encode bits
    ao = np.random.mtrand.RandomState(args["seed"]).permutation(np.arange(args["block_length"]))
    net.model.enc.set_interleaver_order(ao)
    y = net.model.enc(x)

    # transform bits to code
    return Channel.bits_to_sequence(y[0].cpu().detach().numpy(), y.shape)


def decode(args, net, in_seq=None):
    """
    Helper function for "decode".

    :param args: Arguments as dictionary.
    :param net: Instance of the class net.
    :returns: Decoded code.
    """
    # transform code into bits
    if args["bitdec"]:
        inp_ = args["bitdec"]
    else:
        inp_ = in_seq
    if args["rate"] == "onethird":
        dim = 3
    elif args["rate"] == "onehalf":
        dim = 2
    else:
        raise ValueError
    x = Channel.sequence_to_bits(inp_, (1, args["block_length"] + args["block_padding"], dim))
    x = torch.from_numpy(x).cuda() if args["gpu"] else torch.from_numpy(x)
    x = torch.reshape(x, (1, args["block_length"] + args["block_padding"], dim))

    # try to decode code
    ao = np.random.mtrand.RandomState(args["seed"]).permutation(np.arange(args["block_length"]))
    net.model.dec.set_interleaver_order(ao)
    y_t = net.model.coder(x)
    y = net.model.dec(y_t)

    # polish decoded code
    y = torch.round(torch.clamp(y, 0.0, 1.0)).cpu().detach().numpy()
    return "".join(str(e) for e in np.reshape(y, -1).astype("uint8"))
