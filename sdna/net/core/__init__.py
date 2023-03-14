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
    "transformer": EncoderTransformer,
    "cnn_nolat": EncoderCNN_nolat,
    "vae": Encoder_vae,
    "vae_lat": Encoder_vae_lat,
    "gnn": EncoderGNN,
    "cnn_kernel_inc": EncoderCNN_kernel_increase
}

DECODERS = {
    "rnn": DecoderRNN,
    "cnn": DecoderCNN,
    "rnnatt": DecoderRNNatt,
    "transformer": DecoderTransformer,
    "cnn_nolat": DecoderCNN_nolat,
    "entransformer": DecoderEnTransformer,
    "ensemble_dec": EnsembleDecoder
}

CODERS = {
    "mlp": CoderMLP,
    "cnn": CoderCNN,
    "rnn": CoderRNN,
    "transformer": CoderTransformer,
    "cnn_nolat": CoderCNN_nolat,
    "cnn_rnn": CoderCNN_RNN,
    "cnn_conc": CoderCNN_conc,
    "cnn_ensemble": EnsembleCNN,
    "idt": CoderIDT,
    "resnet": ResNetCoder,
    "resnet2d": ResNetCoder2d,
    "resnet2d_1d": ResNetCoder2d_1d
}

__all__ = ['AutoEncoder',
           'Channel',
           'ENCODERS',
           'DECODERS',
           'CODERS']
