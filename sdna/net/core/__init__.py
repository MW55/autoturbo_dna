# -*- coding: utf-8 -*-

# Add classes from directory #
from sdna.net.core.ae import *
from sdna.net.core.channel import *
from sdna.net.core.encoder import *
from sdna.net.core.decoder import *
from sdna.net.core.coder import *
from sdna.net.core.interleaver import *
from sdna.net.core.layers import *

# Add networks with corresponding key #
ENCODERS = {
    "rnn": EncoderRNN,
    "srnn": SysEncoderRNN,
    "cnn": EncoderCNN,
    "scnn": SysEncoderCNN,
    "rnnatt": EncoderRNNatt,
    "transformer": EncoderTransformer
}

DECODERS = {
    "rnn": DecoderRNN,
    "cnn": DecoderCNN,
    "rnnatt": DecoderRNNatt,
    "transformer": DecoderTransformer
}

CODERS = {
    "mlp": CoderMLP,
    "cnn": CoderCNN,
    "rnn": CoderRNN,
    "transformer": CoderTransformer
}

__all__ = ['AutoEncoder',
           'Channel',
           'ENCODERS',
           'DECODERS',
           'CODERS']
