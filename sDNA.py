#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

from sdna.coordinator import SDNA


class SDNAArgs:
    def __init__(self):
        """
        Class is used to correctly parse the arguments for the tool.
        """
        self.parser = argparse.ArgumentParser(
            prog="sDNA",
            description="The tool simulates a DNA data storage using neural networks and a systematic DNA error simulation. The trained models can be used to encode bit streams and decode DNA sequences, while error simulation can be applied to the generated DNA sequences.",
            usage="./sDNA.py [...]")
        self.parser._action_groups.pop()  # create argument groups
        self.parser.add_argument('-v', '--version', action='version', version='%(prog)s v1.0.0')

        # General arguments
        self.generally = self.parser.add_argument_group('General arguments',
                                                        'Note: Only one of the following options can be selected at a time: --train, --encode, --decode, --simulate or --ids.')
        self._generally()

        # NN structure parameters
        self.net_structure = self.parser.add_argument_group('NN structure parameters',
                                                            'Note: If you load a model that has already been created, the structure parameters can no longer be changed, see user guide.')
        self._net_structure()

        # NN training parameters
        self.net_training = self.parser.add_argument_group('NN training parameters')
        self._net_training()

        # DNA synthesis, storage and sequencing simulation
        self.dna_error_simulation = self.parser.add_argument_group('DNA synthesis, storage and sequencing simulation',
                                                                   'Note: Some arguments expect an optional path to a json config/data file, see user guide for usage.')
        self._dna_error_simulation()

        # DNA synthesis, storage and sequencing detection
        self.dna_error_detection = self.parser.add_argument_group('DNA synthesis, storage and sequencing detection')
        self._dna_error_detection()

        self.args = vars(self.parser.parse_args())

    def _generally(self):
        """
        Function to generate the arguments for 'General arguments'.
        """
        self.generally.add_argument('--wdir',
                                    help='Path to the working directory, if not existing the model will be saved here, if already existing the model will be loaded.',
                                    default=None,
                                    metavar='PATH',
                                    dest='working_dir')
        self.generally.add_argument('--train',
                                    help='Create and train the desired model.',
                                    action='store_true',
                                    dest='train')
        self.generally.add_argument('--bitenc',
                                    help='Encode with a model a bit string into a code.',
                                    default=None,
                                    dest='bitenc')
        self.generally.add_argument('--bitdec',
                                    help='Decode with a model a bit string into a code.',
                                    default=None,
                                    dest='bitdec')
        self.generally.add_argument('--encode',
                                    '-e',
                                    help='Encode with a model a file.',
                                    default=None,
                                    action='store_true',
                                    dest='encode')
        self.generally.add_argument('--decode',
                                    '-d',
                                    help='Decode with a model a code back into a file.',
                                    default=None,
                                    action='store_true',
                                    dest='decode')
        self.generally.add_argument('--input',
                                    '-i',
                                    help='Path to the file to be en-/decoded.',
                                    default=None,
                                    type=str,
                                    metavar="PATH",
                                    dest='inp')
        self.generally.add_argument('--output',
                                    '-o',
                                    help='Path to the output file.',
                                    default=None,
                                    type=str,
                                    metavar="PATH",
                                    dest='out')
        self.generally.add_argument('--index_size',
                                    '-is',
                                    help='size (in bits) of the added index, larger files need bigger index sizes, has to be a multiple of 8.',
                                    default=16,
                                    type=int,
                                    dest='index_size')
        self.generally.add_argument('--simulate',
                                    help='Simulate errors on a generated code.',
                                    default=None,
                                    metavar='CODE',
                                    dest='simulate')
        self.generally.add_argument('--ids',
                                    help='Shows a list of the default ids of the different options for DNA synthesis, storage and sequencing simulation.',
                                    action='store_true',
                                    dest='ids')
        self.generally.add_argument('--seed',
                                    help='Specify a integer number, this allows to reproduce the results.',
                                    type=int,
                                    default=0,
                                    metavar='X',
                                    dest='seed')
        self.generally.add_argument('--gpu',
                                    help='Whether the calculations of the models should run on the GPU (using CUDA).',
                                    action='store_true',
                                    dest='gpu')
        self.generally.add_argument('--parallel',
                                    help='Whether to run the calculations on multiple GPUs, if there are more than one.',
                                    action='store_true',
                                    dest='parallel')
        self.generally.add_argument('--threads',
                                    help='If using the CPU, how many threads should be used.',
                                    type=int,
                                    default=8,
                                    dest='threads')

    def _net_structure(self):
        """
        Function to generate the arguments for 'NN structure parameters'.
        """
        self.net_structure.add_argument('--rate',
                                        help='Rate of the code, supported are 1/3 (argument=onethird) and 1/2 (argument=onehalf)',
                                        choices=['onethird', 'onehalf'],
                                        type=str.lower,
                                        metavar='X',
                                        default='onethird',
                                        dest='rate')
        self.net_structure.add_argument('--block-length',
                                        help='Length of the bitstreams to be used. (default=64)',
                                        type=int,
                                        default=64,
                                        metavar='X',
                                        dest='block_length')
        self.net_structure.add_argument('--block-padding',
                                        help='Length of the padding by which the bitstream is extended. (default=18)',
                                        type=int,
                                        default=18,
                                        metavar='X',
                                        dest='block_padding')
        self.net_structure.add_argument('--encoder',
                                        help='Choose which encoder to use: RNN, SRNN, CNN, SCNN or RNNatt. (default=CNN)',
                                        choices=['rnn', 'srnn', 'cnn', 'scnn', 'rnnatt', 'transformer', 'cnn_nolat', 'vae', 'vae_lat', 'gnn', 'cnn_kernel_inc', 'resnet1d'],
                                        type=str.lower,
                                        default='cnn',
                                        metavar='CHOICE',
                                        dest='encoder')
        self.net_structure.add_argument('--enc-units',
                                        help='The number of expected features in the hidden layer for the encoder. (default=64)',
                                        type=int,
                                        default=64,
                                        metavar='X',
                                        dest='enc_units')
        self.net_structure.add_argument('--enc-actf',
                                        help='Choose which activation function should be applied to the encoder: tanh, elu, relu, selu, sigmoid or identity. (default=elu)',
                                        choices=['tanh', 'elu', 'relu', 'selu', 'sigmoid', 'identity', 'leakyrelu', 'gelu'],
                                        type=str.lower,
                                        default='elu',
                                        metavar='CHOICE',
                                        dest='enc_actf')
        self.net_structure.add_argument('--enc-dropout',
                                        help='Dropout probability for the encoder. (default=0.0)',
                                        type=float,
                                        default=0.0,
                                        metavar='X',
                                        dest='enc_dropout')
        self.net_structure.add_argument('--enc-layers',
                                        help='Number of recurrent layers per RNN/CNN structure in the encoder. (default=5)',
                                        type=int,
                                        default=5,
                                        metavar='X',
                                        dest='enc_layers')
        self.net_structure.add_argument('--enc-kernel',
                                        help='Size of the kernels for the CNN in the encoder. (default=5)',
                                        type=int,
                                        default=5,
                                        metavar='X',
                                        dest='enc_kernel')
        self.net_structure.add_argument('--enc-rnn',
                                        help='Choose which structure to use for the RNN in the encoder: GRU or LSTM. (default=GRU)',
                                        choices=['GRU', 'LSTM'],
                                        type=str.upper,
                                        default='GRU',
                                        metavar='CHOICE',
                                        dest='enc_rnn')
        self.net_structure.add_argument('--vae-beta',
                                        help='The beta multiplier of the Kullback–Leibler divergence if using a VAE.',
                                        type=float,
                                        default=0.0,
                                        metavar='X',
                                        dest='beta')
        self.net_structure.add_argument('--decoder',
                                        help='Choose which decoder to use: RNN or CNN. (default=CNN)',
                                        choices=['rnn', 'cnn', 'rnnatt', 'transformer', 'cnn_nolat', 'entransformer', 'ensemble_dec', 'resnet1d'],
                                        type=str.lower,
                                        default='cnn',
                                        metavar='CHOICE',
                                        dest='decoder')
        self.net_structure.add_argument('--dec-units',
                                        help='The number of expected features in the hidden layer for the decoder. (default=64)',
                                        type=int,
                                        default=64,
                                        metavar='X',
                                        dest='dec_units')
        self.net_structure.add_argument('--dec-actf',
                                        help='Choose which activation function should be applied to the decoder: tanh, elu, relu, selu, sigmoid or identity. (default=identity)',
                                        choices=['tanh', 'elu', 'relu', 'selu', 'sigmoid', 'identity', 'leakyrelu', 'gelu'],
                                        type=str.lower,
                                        default='identity',
                                        metavar='CHOICE',
                                        dest='dec_actf')
        self.net_structure.add_argument('--dec-dropout',
                                        help='Dropout probability for the decoder. (default=0.0)',
                                        type=float,
                                        default=0.0,
                                        metavar='X',
                                        dest='dec_dropout')
        self.net_structure.add_argument('--dec-layers',
                                        help='Number of recurrent layers per RNN/CNN structure in the decoder. (default=5)',
                                        type=int,
                                        default=5,
                                        metavar='X',
                                        dest='dec_layers')
        self.net_structure.add_argument('--dec-inputs',
                                        help='The number of expected input features for the decoder. (default=5)',
                                        type=int,
                                        default=5,
                                        metavar='X',
                                        dest='dec_inputs')
        self.net_structure.add_argument('--dec-iterations',
                                        help='Number of iterative loops to be made in the decoder. (default=6)',
                                        type=int,
                                        default=6,
                                        metavar='X',
                                        dest='dec_iterations')
        self.net_structure.add_argument('--dec-kernel',
                                        help='Size of the kernels for the CNN in the decoder. (default=5)',
                                        type=int,
                                        default=5,
                                        metavar='X',
                                        dest='dec_kernel')
        self.net_structure.add_argument('--dec-rnn',
                                        help='Choose which structure to use for the RNN in the decoder: GRU or LSTM. (default=GRU)',
                                        choices=['GRU', 'LSTM'],
                                        type=str.upper,
                                        default='GRU',
                                        metavar='CHOICE',
                                        dest='dec_rnn')
        self.net_structure.add_argument('--not-extrinsic',
                                        help='Whether extrinsic information should be applied to the decoder each iteration. (default=on)',
                                        action='store_false',
                                        dest='extrinsic')
        self.net_structure.add_argument('--coder',
                                        help='Choose which coder to use: MLP, CNN or RNN. (default=CNN)',
                                        choices=['mlp', 'cnn', 'rnn', 'transformer', 'cnn_nolat', 'cnn_rnn', 'cnn_conc', 'cnn_ensemble', "idt", "resnet", "resnet2d", "resnet2d_1d", "resnet_ens", "resnet_sep"],
                                        type=str.lower,
                                        default='cnn',
                                        metavar='CHOICE',
                                        dest='coder')
        self.net_structure.add_argument('--coder-units',
                                        help='The number of expected features in the hidden layer for the coder. (default=64)',
                                        type=int,
                                        default=64,
                                        metavar='X',
                                        dest='coder_units')
        self.net_structure.add_argument('--coder-actf',
                                        help='Choose which activation function should be applied to the coder: tanh, elu, relu, selu, sigmoid or identity. (default=elu)',
                                        choices=['tanh', 'elu', 'relu', 'selu', 'sigmoid', 'identity', 'leakyrelu', 'gelu'],
                                        type=str.lower,
                                        default='elu',
                                        metavar='CHOICE',
                                        dest='coder_actf')
        self.net_structure.add_argument('--coder-dropout',
                                        help='Dropout probability for the coder. (default=0.0)',
                                        type=float,
                                        default=0.0,
                                        metavar='X',
                                        dest='coder_dropout')
        self.net_structure.add_argument('--coder-layers',
                                        help='Number of recurrent layers per RNN/CNN structure in the coder. (default=5)',
                                        type=int,
                                        default=5,
                                        metavar='X',
                                        dest='coder_layers')
        self.net_structure.add_argument('--coder-kernel',
                                        help='Size of the kernels for the CNN in the coder. (default=5)',
                                        type=int,
                                        default=5,
                                        metavar='X',
                                        dest='coder_kernel')
        self.net_structure.add_argument('--coder-rnn',
                                        help='Choose which structure to use for the RNN in the coder: GRU or LSTM. (default=GRU)',
                                        choices=['GRU', 'LSTM'],
                                        type=str.upper,
                                        default='GRU',
                                        metavar='CHOICE',
                                        dest='coder_rnn')
        self.net_structure.add_argument('--init-weights',
                                        help='Choose which method to use to initialize the linear layers of the model: normal, uniform, constant, xavier_normal, xavier_uniform, kaiming_normal or kaiming_uniform. (default=custom)',
                                        choices=['normal', 'uniform', 'constant', 'xavier_normal', 'xavier_uniform',
                                                 'kaiming_normal', 'kaiming_uniform'],
                                        type=str.lower,
                                        default=None,
                                        metavar='CHOICE',
                                        dest='init_weights')
        self.net_structure.add_argument('--lat-redundancy',
                                        help='Redundancy of the final encoder layer (and first decoder layer), required to account for constraints. Has to be divisible by 2',
                                        type=int,
                                        default=0,
                                        metavar='X',
                                        dest='redundancy')
        self.net_structure.add_argument('--ens_models',
                                        help='If ensemble coders are used, defines the number of coder instances in the ensemble.',
                                        type=int,
                                        default=3,
                                        metavar='X',
                                        dest='n_models')
    def _net_training(self):
        """
        Function to generate the arguments for 'NN training parameters'.
        """
        self.net_training.add_argument('--blocks',
                                       help='Number of the bitstreams to be used. (default=1024)',
                                       type=int,
                                       default=1024,
                                       metavar='X',
                                       dest='blocks')
        self.net_training.add_argument('--batch-size',
                                       help='Size of the batch to be used during training. (default=256)',
                                       type=int,
                                       default=256,
                                       metavar='X',
                                       dest='batch_size')
        self.net_training.add_argument('--epochs',
                                       help='Number of epochs the whole model should be trained. (default=100)',
                                       type=int,
                                       default=100,
                                       metavar='X',
                                       dest='epochs')
        self.net_training.add_argument('--enc-lr',
                                       help='Value of the learning rate to be used for the encoder. (default=0.00001)',
                                       type=float,
                                       default=0.00001,
                                       metavar='X',
                                       dest='enc_lr')
        self.net_training.add_argument('--enc-optimizer',
                                       help='Choose which optimizer to use for the encoder: Adam, SGD or Adagrad. (default=Adam)',
                                       choices=['adam', 'sgd', 'adagrad'],
                                       type=str.lower,
                                       default='adam',
                                       metavar='CHOICE',
                                       dest='enc_optimizer')
        self.net_training.add_argument('--enc-steps',
                                       help='Number of training steps to be performed per epoch for the encoder. (default=1)',
                                       type=int,
                                       default=1,
                                       metavar='X',
                                       dest='enc_steps')
        self.net_training.add_argument('--dec-lr',
                                       help='Value of the learning rate to be used for the decoder. (default=0.00001)',
                                       type=float,
                                       default=0.00001,
                                       metavar='X',
                                       dest='dec_lr')
        self.net_training.add_argument('--dec-optimizer',
                                       help='Choose which optimizer to use for the decoder: Adam, SGD or Adagrad. (default=Adam)',
                                       choices=['adam', 'sgd', 'adagrad'],
                                       type=str.lower,
                                       default='adam',
                                       metavar='CHOICE',
                                       dest='dec_optimizer')
        self.net_training.add_argument('--dec-steps',
                                       help='Number of training steps to be performed per epoch for the decoder. (default=2)',
                                       type=int,
                                       default=2,
                                       metavar='X',
                                       dest='dec_steps')
        self.net_training.add_argument('--coder-lr',
                                       help='Value of the learning rate to be used for the coder. (default=0.001)',
                                       type=float,
                                       default=0.001,
                                       metavar='X',
                                       dest='coder_lr')
        self.net_training.add_argument('--coder-optimizer',
                                       help='Choose which optimizer to use for the coder: Adam, SGD or Adagrad. (default=Adam)',
                                       choices=['adam', 'sgd', 'adagrad'],
                                       type=str.lower,
                                       default='adam',
                                       metavar='CHOICE',
                                       dest='coder_optimizer')
        self.net_training.add_argument('--coder-steps',
                                       help='Number of training steps to be performed per epoch for the coder. (default=5)',
                                       type=int,
                                       default=5,
                                       metavar='X',
                                       dest='coder_steps')
        self.net_training.add_argument('--simultaneously',
                                       help='Whether the encoder and decoder are to be trained at the same time, if so, the learning parameters from the encoder are used. (default=off)',
                                       action='store_true',
                                       dest='simultaneously_training')
        self.net_training.add_argument('--batch-norm',
                                       help='Whether to use batch normalization or not.',
                                       type=bool,
                                       default=False,
                                       metavar='X',
                                       dest='batch_norm')
        self.net_training.add_argument('--separate-coder-training',
                                       help="If the coder should be split into 3 seperate instances during training.",
                                       action='store_true',
                                       dest='separate_coder_training')

    def _dna_error_simulation(self):
        """
        Function to generate the arguments for 'DNA synthesis, storage and sequencing simulation'.
        """
        self.dna_error_simulation.add_argument('--synthesis',
                                               help='Specify the id of the synthesis method. (default=ErrASE)',
                                               nargs='+',
                                               default=('1', None),
                                               metavar=('ID', '\b[PATH]\x1b['),
                                               dest='synthesis')
        self.dna_error_simulation.add_argument('--pcr-cycles',
                                               help='Number of cycles to be used for the PCR. (default=30)',
                                               type=int,
                                               default=30,
                                               metavar='X',
                                               dest='pcr_cycles')
        self.dna_error_simulation.add_argument('--pcr',
                                               help='Specify the id of the PCR type. (default=Taq)',
                                               nargs='+',
                                               default=('14', None),
                                               metavar=('ID', '\b[PATH]\x1b['),
                                               dest='pcr')
        self.dna_error_simulation.add_argument('--storage-months',
                                               help='Months of storage to be simulated. (default=24)',
                                               type=int,
                                               default=24,
                                               metavar='X',
                                               dest='storage_months')
        self.dna_error_simulation.add_argument('--storage',
                                               help='Specify the id of the storage host. (default=E coli)',
                                               nargs='+',
                                               default=('1', None),
                                               metavar=('ID', '\b[PATH]\x1b['),
                                               dest='storage')
        self.dna_error_simulation.add_argument('--sequencing',
                                               help='Specify the id of the sequencing method. (default=Paired End)',
                                               nargs='+',
                                               default=('2', None),
                                               metavar=('ID', '\b[PATH]\x1b['),
                                               dest='sequencing')
        self.dna_error_simulation.add_argument('--amplifier',
                                               help='Value by how much more distinct the errors should be. (default=5.0)',
                                               type=float,
                                               default=5.0,
                                               metavar='X',
                                               dest='amplifier')

    def _dna_error_detection(self):
        """
        Function to generate the arguments for 'DNA synthesis, storage and sequencing detection'.
        """
        self.dna_error_detection.add_argument('--probabilities',
                                              help='Path to json file for error probabilities. (optional)',
                                              default='config/error_detection/probabilities.json',
                                              metavar='PATH',
                                              dest='probabilities_json')
        self.dna_error_detection.add_argument('--useq',
                                              help='Path to json file for undesired sequences. (optional)',
                                              default='config/error_detection/undesired_sequences.json',
                                              metavar='PATH',
                                              dest='useq_json')
        self.dna_error_detection.add_argument('--gc-window',
                                              help='Size of the window to be used for the GC-Content error probability detection. (default=50)',
                                              type=int,
                                              default=50,
                                              metavar='X',
                                              dest='gc_window')
        self.dna_error_detection.add_argument('--kmer-window',
                                              help='Size of the window to be used for the Kmer error probability detection. (default=10)',
                                              type=int,
                                              default=10,
                                              metavar='X',
                                              dest='kmer_window')


if __name__ == '__main__':
    SDNA(SDNAArgs().args)
