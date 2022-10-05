# sDNA: simulated deep learning-based DNA data storage system

The tool **sDNA** simulates a DNA data storage using neural networks and a systematic DNA synthesis, storage and sequencing simulation. The trained models can be used to encode bit streams into DNA sequences and decode DNA sequences back to bit streams, while error simulation can be applied to the generated DNA sequences.

## Installation:
Install Python 3.7.x if not already done (if the GPU should be used, a CUDA-capable system is required and the corresponding dependencies). Clone or download this repository, and install the dependencies:
```bash
git clone https://github.com/MW55/autoturbo_dna.git

# install packages under python 3.7.x
pip install torch numpy scipy regex
```
## Useage
Train (Simple): --wdir models/simple_train/ --train Encode (Simple): --wdir models/simple_train/ -i test_data/MOSLA.txt -o test_data/MOSLA_encoded.fasta -e Decode (Simple): --wdir models/simple_train/ -i test_data/MOSLA_encoded.fasta -o test_data/MOSLA_decoded.txt -d

-e: encode -d: decode --train: train --wdir: relative path to the model --threads: number of threads

there are quite a lot more optional parameters, you can find an overview and explainations in sDNA.py in the root directory.

## Usage (Deprecated):
There are two pre-trained models in the repository that can be used (one for the CPU, the other for the GPU, accuracy of the models is ~99.20%):

(1) Encode the bit stream into a DNA sequence.
```bash
./sDNA.py --wdir models/cnn-cpu/ --bitenc 0110100001100101011011000110110001101111001000000111100101101111
> GTTAGTGGGTCAGTCAGTCCATAAGCTGGTCCCATCCACGGATGCATGCATAGTCCCAACGATGCGAGTGGGATCCTGAATCGTAGAATCTATGCC
```
(2) Apply the DNA error simulation to the DNA sequence.
```bash
./sDNA.py --simulate GTTAGTGGGTCAGTCAGTCCATAAGCTGGTCCCATCCACGGATGCATGCATAGTCCCAACGATGCGAGTGGGATCCTGAATCGTAGAATCTATGCC
> GTTAGTGGGTCAGTCAGTCCATAAGCTGGTCCCATCCACGGATGTATGCATAGTCCCAACGATGGGAGTGGGATCCTGAATCGTAGAATCTATGCC
```
(3) Decode the DNA sequence back into the bit stream.
```bash
./sDNA.py --wdir models/cnn-cpu/ --decode GTTAGTGGGTCAGTCAGTCCATAAGCTGGTCCCATCCACGGATGTATGCATAGTCCCAACGATGGGAGTGGGATCCTGAATCGTAGAATCTATGCC
> 0110100001100101011011000110110001101111001000000111100101101111
```

There are over 51 different arguments that influence the model and with which the specific DNA data storage can be defined. The training of a model can be canceled at any time and continued at a later point. A call with the default values to train a model could look like this and be extended by certain arguments:
```bash
./sDNA.py --wdir models/example/ --train --gpu [...]
```
