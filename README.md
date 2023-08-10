# Autoturbo-DNA: Turbo-Autoencoders for the DNA data storage channel

**Autoturbo-DNA** is a comprehensive autoencoder framework designed for the specific challenges of DNA data storage. It leverages the principles of Turbo Autoencoders (Jiang et. al., 2019) while integrating critical components for DNA data storage applications.

### Key Features:

- **End-to-End Integration**: Combines Turbo Autoencoder principles with DNA data storage channel simulation.
- **Modular Architecture**: Supports a wide range of Neural Network architectures which can be easily integrated.
- **Configurable Components**: Components can be customized using a configuration file.
- **Flexible Parameter Adjustments**: User-centric design allows for easy adjustments of DNA data storage channel settings and constraint adherence parameters.


## Installation:
Install Python 3.7.x if not already done (if the GPU should be used, a CUDA-capable system is required and the corresponding dependencies). Clone or download this repository, and install the dependencies:
```bash
git clone https://github.com/MW55/autoturbo_dna.git

# install packages under python 3.7.x
pip install torch numpy scipy regex
```
## Useage
Train: 

```bash
./sDNA.py --wdir models/simple_train/ --train 
```

During training, a config will be generated in the model folder containing all the additional parameters used.


Encode: 

```bash
./sDNA.py --wdir models/simple_train/ -i test_data/MOSLA.txt -o test_data/MOSLA_encoded.fasta -e 
```

Decode: 

```bash
./sDNA.py --wdir models/simple_train/ -i test_data/MOSLA_encoded.fasta -o test_data/MOSLA_decoded.txt -d
```
## Configuration:

There are different arguments that influence the model and with which the specific DNA data storage can be defined. The training of a model can be canceled at any time and continued at a later point. A call with the default values to train a model could look like this and be extended by certain arguments:

| Option Strings            | Type  | Default                   | Help                                                                                                                                                                     |
|---------------------------|-------|---------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| -h, --help                | None  | ==SUPPRESS==              | show this help message and exit                                                                                                                                          |
| -v, --version             | None  | ==SUPPRESS==              | show program's version number and exit                                                                                                                                   |
| --wdir                    | str   | None                      | Path to the working directory, if not existing the model will be saved here, if already existing the model will be loaded.                                               |
| --train                   | bool  | False                     | Create and train the desired model.                                                                                                                                      |
| --bitenc                  | None  | None                      | Encode with a model a bit string into a code.                                                                                                                            |
| --bitdec                  | None  | None                      | Decode with a model a bit string into a code.                                                                                                                            |
| --encode, -e              | None  | None                      | Encode with a model a file.                                                                                                                                              |
| --decode, -d              | None  | None                      | Decode with a model a code back into a file.                                                                                                                             |
| --input, -i               | str   | None                      | Path to the file to be en-/decoded.                                                                                                                                      |
| --output, -o              | str   | None                      | Path to the output file.                                                                                                                                                 |
| --index\_size, -is        | int   | 16                        | size (in bits) of the added index, larger files need bigger index sizes, has to be a multiple of 8.                                                                      |
| --simulate                | None  | None                      | Simulate errors on a generated code.                                                                                                                                     |
| --ids                     | None  | False                     | Shows a list of the default ids of the different options for DNA synthesis, storage and sequencing simulation.                                                           |
| --seed                    | int   | 0                         | Specify a integer number, this allows to reproduce the results.                                                                                                          |
| --gpu                     | None  | False                     | Whether the calculations of the models should run on the GPU (using CUDA).                                                                                               |
| --parallel                | None  | False                     | Whether to run the calculations on multiple GPUs, if there are more than one.                                                                                            |
| --threads                 | int   | 8                         | If using the CPU, how many threads should be used.                                                                                                                       |
| --rate                    | str   | onethird                  | Rate of the code, supported are 1/3 (argument=onethird) and 1/2 (argument=onehalf)                                                                                       |
| --block-length            | int   | 64                        | Length of the bitstreams to be used                                                                                                                                      |
| --block-padding           | int   | 18                        | Length of the padding by which the bitstream is extended                                                                                                                 |
| --encoder                 | str   | cnn                       | Choose which encoder to use: RNN, SRNN, CNN, SCNN or RNNatt                                                                                                              |
| --enc-units               | int   | 64                        | The number of expected features in the hidden layer for the encoder                                                                                                      |
| --enc-actf                | str   | elu                       | Choose which activation function should be applied to the encoder: tanh, elu, relu, selu, sigmoid or identity                                                            |
| --enc-dropout             | float | 0.0                       | Dropout probability for the encoder                                                                                                                                      |
| --enc-layers              | int   | 5                         | Number of recurrent layers per RNN/CNN structure in the encoder                                                                                                          |
| --enc-kernel              | int   | 5                         | Size of the kernels for the CNN in the encoder                                                                                                                           |
| --enc-rnn                 | str   | GRU                       | Choose which structure to use for the RNN in the encoder: GRU or LSTM                                                                                                    |
| --vae-beta                | float | 0.0                       | The beta multiplier of the Kullback–Leibler divergence if using a VAE.                                                                                                   |
| --decoder                 | str   | cnn                       | Choose which decoder to use: RNN or CNN                                                                                                                                  |
| --dec-units               | int   | 64                        | The number of expected features in the hidden layer for the decoder                                                                                                      |
| --dec-actf                | str   | identity                  | Choose which activation function should be applied to the decoder: tanh, elu, relu, selu, sigmoid or identity                                                            |
| --dec-dropout             | float | 0.0                       | Dropout probability for the decoder                                                                                                                                      |
| --dec-layers              | int   | 5                         | Number of recurrent layers per RNN/CNN structure in the decoder                                                                                                          |
| --dec-inputs              | int   | 5                         | The number of expected input features for the decoder                                                                                                                    |
| --dec-iterations          | int   | 6                         | Number of iterative loops to be made in the decoder                                                                                                                      |
| --dec-kernel              | int   | 5                         | Size of the kernels for the CNN in the decoder                                                                                                                           |
| --dec-rnn                 | str   | GRU                       | Choose which structure to use for the RNN in the decoder: GRU or LSTM                                                                                                    |
| --not-extrinsic           | None  | True                      | Whether extrinsic information should be applied to the decoder each iteration                                                                                            |
| --coder                   | str   | cnn                       | Choose which coder to use: MLP, CNN or RNN                                                                                                                               |
| --coder-units             | int   | 64                        | The number of expected features in the hidden layer for the coder                                                                                                        |
| --coder-actf              | str   | elu                       | Choose which activation function should be applied to the coder: tanh, elu, relu, selu, sigmoid or identity                                                              |
| --coder-dropout           | float | 0.0                       | Dropout probability for the coder                                                                                                                                        |
| --coder-layers            | int   | 5                         | Number of recurrent layers per RNN/CNN structure in the coder                                                                                                            |
| --coder-kernel            | int   | 5                         | Size of the kernels for the CNN in the coder                                                                                                                             |
| --coder-rnn               | str   | GRU                       | Choose which structure to use for the RNN in the coder: GRU or LSTM                                                                                                      |
| --init-weights            | str   | None                      | Choose which method to use to initialize the linear layers of the model: normal, uniform, constant, xavier\_normal, xavier\_uniform, kaiming\_normal or kaiming\_uniform |
| --lat-redundancy          | int   | 0                         | Redundancy of the final encoder layer (and first decoder layer), required to account for constraints. Has to be divisible by 2                                           |
| --ens-models              | int   | 3                         | If ensemble coders are used, defines the number of coder instances in the ensemble.                                                                                      |
| --padding-style           | str   | constant                  | If padding should be constant values or a circular copy of the input.                                                                                                    |
| --blocks                  | int   | 1024                      | Number of the bitstreams to be used                                                                                                                                      |
| --batch-size              | int   | 256                       | Size of the batch to be used during training                                                                                                                             |
| --epochs                  | int   | 100                       | Number of epochs the whole model should be trained                                                                                                                       |
| --enc-lr                  | float | 0.00001                   | Value of the learning rate to be used for the encoder                                                                                                                    |
| --enc-optimizer           | str   | adam                      | Choose which optimizer to use for the encoder: Adam, SGD or Adagrad                                                                                                      |
| --enc-steps               | int   | 1                         | Number of training steps to be performed per epoch for the encoder                                                                                                       |
| --dec-lr                  | float | 0.00001                   | Value of the learning rate to be used for the decoder                                                                                                                    |
| --dec-optimizer           | str   | adam                      | Choose which optimizer to use for the decoder: Adam, SGD or Adagrad                                                                                                      |
| --dec-steps               | int   | 2                         | Number of training steps to be performed per epoch for the decoder                                                                                                       |
| --coder-lr                | float | 0.001                     | Value of the learning rate to be used for the coder                                                                                                                      |
| --coder-optimizer         | str   | adam                      | Choose which optimizer to use for the coder: Adam, SGD or Adagrad                                                                                                        |
| --coder-steps             | int   | 5                         | Number of training steps to be performed per epoch for the coder                                                                                                         |
| --simultaneously          | None  | False                     | Whether the encoder and decoder are to be trained at the same time, if so, the learning parameters from the encoder are used                                             |
| --batch-norm              | bool  | False                     | Whether to use batch normalization or not.                                                                                                                               |
| --separate-coder-training | None  | False                     | If the coder should be split into 3 seperate instances during training.                                                                                                  |
| --all-errors              | None  | False                     | train each part of the model always with all error types.                                                                                                                |
| --channel                 | str   | dna                       | which channel model should be used for training                                                                                                                          |
| --continuous-coder        | None  | False                     | toggles that the intermediate decoder (coder) passes continuous values to the decoder.                                                                                   |
| --constraint-training     | None  | False                     | If the code should also be trained to adhere to constraints.                                                                                                             |
| --loss-beta               | float | 1.0                       | beta parameter for the smooth L1 loss.                                                                                                                                   |
| --coder-train-target      | str   | encoded\_data             | how the coder should be trained, for best reconstruction accuracy or to be as close to the encoder output as possible                                                    |
| --simultaneously-warmup   | int   | 0                         | if using simultaneously training, how many warmup epochs should be trained seperatly, before moving to simultaneously training.                                          |
| --synthesis               | None  | (1, None)                 | Specify the id of the synthesis method                                                                                                                                   |
| --pcr-cycles              | int   | 30                        | Number of cycles to be used for the PCR                                                                                                                                  |
| --pcr                     | None  | (14, None)                | Specify the id of the PCR type                                                                                                                                           |
| --storage-months          | int   | 24                        | Months of storage to be simulated                                                                                                                                        |
| --storage                 | None  | (1, None)                 | Specify the id of the storage host                                                                                                                                       |
| --sequencing              | None  | (2, None)                 | Specify the id of the sequencing method                                                                                                                                  |
| --amplifier               | float | 5.0                       | Value by how much more distinct the errors should be                                                                                                                     |
| --probabilities           | str   | probabilities.json        | Path to json file for error probabilities                                                                                                                                |
| --useq                    | str   | undesired\_sequences.json | Path to json file for undesired sequences                                                                                                                                |
| --gc-window               | int   | 50                        | Size of the window to be used for the GC-Content error probability detection                                                                                             |
| --kmer-window             | int   | 10                        | Size of the window to be used for the Kmer error probability detection                                                                                                   |

[1]:  Y. Jiang, H. Kim, H. Asnani, S. Kannan, S. Oh, and P. Viswanath, “Turbo autoencoder: Deep learning based channel codes for point-to-point communication channels,” in Advances in Neural Information Processing Systems, pp. 2754–2764, 2019.
