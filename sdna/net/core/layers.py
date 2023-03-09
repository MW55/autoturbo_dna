# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as func


class Quantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs):
        """
        Performs the forward operation.

        :param ctx: Autograd object with helper functions.
        :param inputs: Input tensors.
        :return: Output tensors.
        """
        ctx.save_for_backward(inputs)   # save tensors for backward function

        x_norm = torch.clamp(inputs, -1.0, 1.0)
        x = torch.sign(x_norm)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        """
        Applies gradient formula.

        :param ctx: Autograd object with helper functions.
        :param grad_output: Input tensors.
        :return: Output tensors.
        """
        inputs, = ctx.saved_tensors     # get saved tensors from forward function

        grad_output[inputs > 1.0] = 0
        grad_output[inputs < -1.0] = 0
        grad_output = torch.clamp(grad_output, -0.01, 0.01)
        x = grad_output.clone()
        return x, None


class Conv1d(torch.nn.Module):
    def __init__(self, actf, layers, in_channels, out_channels, kernel_size):
        """
        Applies multiple 1D convolution over an input tensor composed of several input planes.

        :param actf: The activating function to be applied.
        :param layers: The number of convolution to be used.
        :param in_channels: Number of units in the input.
        :param out_channels: Number of output channel produced by the convolution.
        :param kernel_size: Size of the convolving kernel.
        """
        super(Conv1d, self).__init__()
        self._actf = actf
        self._layers = layers

        self._cnns = torch.nn.ModuleList()

        for i in range(self._layers):
            if i == 0:
                self._cnns.append(torch.nn.Conv1d(in_channels=in_channels,
                                                  out_channels=out_channels,
                                                  kernel_size=kernel_size,
                                                  stride=1,
                                                  padding=(kernel_size // 2),
                                                  dilation=1,
                                                  groups=1,
                                                  bias=True))
            else:
                self._cnns.append(torch.nn.Conv1d(in_channels=out_channels,
                                                  out_channels=out_channels,
                                                  kernel_size=kernel_size,
                                                  stride=1,
                                                  padding=(kernel_size // 2),
                                                  dilation=1,
                                                  groups=1,
                                                  bias=True))

    def actf(self, inputs):
        """
        Activation functions which is called from forward function.

        :param inputs: Input tensors.
        :return: Output tensor to which the activation function is applied.
        """
        if self._actf == "tanh":
            return torch.tanh(inputs)
        if self._actf == "elu":
            return func.elu(inputs)
        elif self._actf == "relu":
            return torch.relu(inputs)
        elif self._actf == "selu":
            return func.selu(inputs)
        elif self._actf == "sigmoid":
            return torch.sigmoid(inputs)
        elif self._actf == "identity":
            return inputs
        elif self._actf == "leakyrelu":
            return func.leaky_relu(inputs)
        elif self._actf == "gelu":
            return func.gelu(inputs)
        else:
            return inputs

    def forward(self, inputs):
        """
        Calculates output tensors from input tensors based on the process.

        :param inputs: Input tensor.
        :return: Output tensor of encoder.
        """
        inputs = torch.transpose(inputs, 1, 2)
        x_t = inputs
        for i in range(self._layers):
            x_t = self.actf(self._cnns[i](x_t))

        x = torch.transpose(x_t, 1, 2)
        return x

class Conv1d_inc_kernel(torch.nn.Module):
    def __init__(self, actf, layers, in_channels, out_channels, kernel_size):
        """
        Applies multiple 1D convolution over an input tensor composed of several input planes.

        :param actf: The activating function to be applied.
        :param layers: The number of convolution to be used.
        :param in_channels: Number of units in the input.
        :param out_channels: Number of output channel produced by the convolution.
        :param kernel_size: Size of the convolving kernel.
        """
        super(Conv1d_inc_kernel, self).__init__()
        self._actf = actf
        self._layers = layers

        self._cnns = torch.nn.ModuleList()

        for i in range(self._layers):
            if i == 0:
                self._cnns.append(torch.nn.Conv1d(in_channels=in_channels,
                                                  out_channels=out_channels,
                                                  kernel_size=(kernel_size//2),
                                                  stride=1,
                                                  padding=(kernel_size//3),
                                                  dilation=1,
                                                  groups=1,
                                                  bias=True))
            elif i < self._layers//2:
                self._cnns.append(torch.nn.Conv1d(in_channels=out_channels,
                                                  out_channels=out_channels,
                                                  kernel_size=(kernel_size - 1) // 2,
                                                  stride=1,
                                                  padding=(kernel_size//4),
                                                  dilation=1,
                                                  groups=1,
                                                  bias=True))
            else:
                self._cnns.append(torch.nn.Conv1d(in_channels=out_channels,
                                                  out_channels=out_channels,
                                                  kernel_size=kernel_size,
                                                  stride=1,
                                                  padding=(kernel_size - 1) // 2,
                                                  dilation=1,
                                                  groups=1,
                                                  bias=True))

    def actf(self, inputs):
        """
        Activation functions which is called from forward function.

        :param inputs: Input tensors.
        :return: Output tensor to which the activation function is applied.
        """
        if self._actf == "tanh":
            return torch.tanh(inputs)
        if self._actf == "elu":
            return func.elu(inputs)
        elif self._actf == "relu":
            return torch.relu(inputs)
        elif self._actf == "selu":
            return func.selu(inputs)
        elif self._actf == "sigmoid":
            return torch.sigmoid(inputs)
        elif self._actf == "identity":
            return inputs
        elif self._actf == "leakyrelu":
            return func.leaky_relu(inputs)
        elif self._actf == "gelu":
            return func.gelu(inputs)
        else:
            return inputs

    def forward(self, inputs):
        """
        Calculates output tensors from input tensors based on the process.

        :param inputs: Input tensor.
        :return: Output tensor of encoder.
        """
        inputs = torch.transpose(inputs, 1, 2)
        x_t = inputs
        for i in range(self._layers):
            x_t = self.actf(self._cnns[i](x_t))

        x = torch.transpose(x_t, 1, 2)
        return x



class ForbiddenSeqActivation(torch.nn.Module):
    def __init__(self, window_size=8):
        super(ForbiddenSeqActivation, self).__init__()
        # Define the forbidden sequences
        self.forbidden_seqs = [
            [-1, -1, -1, -1, -1, -1, -1, -1],
            [-1, 1, -1, 1, -1, 1, -1, 1],
            [1, -1, 1, -1, 1, -1, 1, -1],
            [1, 1, 1, 1, 1, 1, 1, 1]
        ]
        self.window_size = window_size

    def forward(self, x):
        # Convert the input tensor to a list of sequences
        input_seqs = x.tolist()
        # Iterate over each sequence and apply sliding window
        for i, seq in enumerate(input_seqs):
            for j in range(len(seq) - self.window_size + 1):
                window = seq[j:j+self.window_size]
                for forbidden_seq in self.forbidden_seqs:
                    if window == forbidden_seq:
                        # Replace the last value in the forbidden subsequence with its opposite sign
                        last_val = window[-1]
                        new_last_val = -last_val
                        input_seqs[i][j+self.window_size-1] = new_last_val
        # Convert the modified input back to a tensor and apply the ReLU activation function
        output_tensor = torch.tensor(input_seqs)
        output_tensor = torch.relu(output_tensor)
        return output_tensor

'''
class IDTLayer(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(IDTLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding layer
        self.embedding = torch.nn.Embedding(input_size, hidden_size)

        # Positional encoding layer
        self.positional_encoding = torch.nn.Embedding(output_size, hidden_size)

        # Transformer layers
        self.transformer_layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(hidden_size, nhead=8) #8 = hardcoded seqlen???
            for _ in range(num_layers)
        ])

        # Output layer
        self.output_layer = torch.nn.Linear(hidden_size, input_size)

        # Copy probabilities layer
        self.copy_layer = torch.nn.Linear(hidden_size, 2)

    def forward(self, input_seq, target_seq):
        # Embed input sequence
        embedded_input = self.embedding(input_seq)

        # Generate target sequence mask
        target_mask = (target_seq != 0).unsqueeze(-2)

        # Embed target sequence and add positional encoding
        embedded_target = self.embedding(target_seq) + self.positional_encoding.weight

        # Apply transformer layers
        transformed = embedded_target
        for transformer_layer in self.transformer_layers:
            transformed = transformer_layer(transformed, target_mask)

        # Generate copy probabilities
        copy_probabilities = self.copy_layer(transformed)
        copy_probabilities = func.softmax(copy_probabilities, dim=-1)

        # Generate output sequence
        output = self.output_layer(transformed)
        output_probabilities = func.softmax(output, dim=-1)

        # Combine copy and output probabilities
        output_probabilities = copy_probabilities[..., 0].unsqueeze(-1) * output_probabilities
        copy_probabilities = copy_probabilities[..., 1].unsqueeze(-1) * embedded_input
        output_probabilities += copy_probabilities

        return output_probabilities
'''

class IDTLayer(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, nhead):
        super(IDTLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nhead = nhead

        # Embedding layer
        self.embedding = torch.nn.Embedding(input_size, hidden_size)

        # Positional encoding layer
        self.positional_encoding = torch.nn.Embedding(output_size, hidden_size)

        # Transformer layers
        self.transformer_layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(hidden_size, nhead=self.nhead)
            for _ in range(num_layers)
        ])

        # Output layer
        self.output_layer = torch.nn.Linear(hidden_size, input_size)

        # Copy probabilities layer
        self.copy_layer = torch.nn.Linear(hidden_size, 2)

    def forward(self, input_seq, target_seq):
        # Pad input sequence tensor
        #input_seq = input_seq.reshape(input_seq.size(0), -1)
        #target_seq = target_seq.reshape(target_seq.size(0), -1)

        batch_size, seq_len, hidden_size = input_seq.shape

        # create attention mask tensor
        #attn_mask = torch.triu(torch.ones(seq_len, seq_len) * -1e9, diagonal=1)
        #attn_mask = attn_mask.unsqueeze(0).repeat(batch_size, 1, 1)

        attn_mask = torch.triu(torch.ones(seq_len, seq_len) * -1e9, diagonal=1).unsqueeze(0)
        attn_mask = attn_mask.repeat(batch_size, 1, 1)


        input_seq_padded = torch.nn.utils.rnn.pad_sequence(input_seq, batch_first=True)

        # Embed input sequence
        input_seq_padded = torch.where(input_seq_padded == -1, torch.tensor([0]), input_seq_padded).long()
        target_seq = torch.where(target_seq == -1, torch.tensor([0]), target_seq).long()

        batch_size, seq_len, individual_bits = input_seq.shape

        # Reshape input_seq and target_seq
        input_seq = input_seq.view(batch_size, seq_len * individual_bits)
        target_seq = target_seq.view(batch_size, seq_len * individual_bits)

        embedded_input = self.embedding(input_seq_padded)

        # Generate target sequence mask
        #target_mask = (target_seq != 0).unsqueeze(-2)

        # Embed target sequence and add positional encoding
        embedded_target = self.embedding(target_seq) + self.positional_encoding.weight

        # Apply transformer layers
        transformed = embedded_target
        for transformer_layer in self.transformer_layers:
            transformed = transformer_layer(transformed, attn_mask)

        # Generate copy probabilities
        copy_probabilities = self.copy_layer(transformed)
        copy_probabilities = func.softmax(copy_probabilities, dim=-1)

        # Generate output sequence
        output = self.output_layer(transformed)
        output_probabilities = func.softmax(output, dim=-1)

        # Combine copy and output probabilities
        output_probabilities = copy_probabilities[..., 0].unsqueeze(-1) * output_probabilities
        copy_probabilities = copy_probabilities[..., 1].unsqueeze(-1) * embedded_input
        output_probabilities += copy_probabilities

        # Reshape output_probabilities
        output_probabilities = output_probabilities.view(batch_size, seq_len, self.input_size)


        # Remove padding from output sequence tensor
        output_probabilities = torch.nn.utils.rnn.pack_padded_sequence(output_probabilities, lengths=input_seq.ne(0).sum(dim=-1), batch_first=True, enforce_sorted=False).data

        output_probabilities = torch.where(output_probabilities == 0, torch.tensor([-1]), output_probabilities).float()
        return output_probabilities

'''
class IDTLayer(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input embedding layer
        self.embedding = torch.nn.Embedding(2, hidden_size, padding_idx=0)

        # Encoder and decoder layers
        self.encoder = torch.nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.decoder = torch.nn.LSTMCell(hidden_size, hidden_size)

        # Output linear layer
        self.linear = torch.nn.Linear(hidden_size, 2)

    def forward(self, input_seq, seq_length):
        batch_size, seqlen, individual_bits = input_seq.size()
        input_seq = input_seq.view(batch_size, -1)
        batch_size, max_seq_length = input_seq.size()

        # Embed input sequence
        input_seq = torch.where(input_seq == -1, torch.tensor([0]), input_seq).long()
        input_embed = self.embedding(input_seq)

        # Initialize hidden state of encoder
        h_0 = torch.zeros(1, batch_size, self.hidden_size).to(input_seq.device)
        c_0 = torch.zeros(1, batch_size, self.hidden_size).to(input_seq.device)

        # Encode input sequence with LSTM
        encoder_output, (h_n, c_n) = self.encoder(input_embed, (h_0, c_0))

        # Initialize hidden state of decoder
        decoder_h = h_n[-1]
        decoder_c = torch.zeros_like(decoder_h)

        # Initialize output sequence
        output_seq = torch.zeros(batch_size, max_seq_length, 2).to(input_seq.device)

        # Loop over each element in the sequence
        for t in range(max_seq_length):
            # Predict next token
            decoder_h, decoder_c = self.decoder(encoder_output[:, t, :], (decoder_h, decoder_c))
            output_logits = self.linear(decoder_h)
            output_probs = torch.softmax(output_logits, dim=-1)

            # Insertion step
            if t < seq_length:
                output_seq[:, t, :] = output_probs
            # Deletion step
            else:
                output_seq[:, t-1, :] = output_probs

        #batch_size, seq_len_mul_ind_bits, _ = output_seq.size()
        #output_seq = output_seq.view(batch_size, seqlen, individual_bits, 2).permute(0, 1, 3, 2).reshape(batch_size, seqlen,
        #                                                                                    individual_bits)
        batch_size, seq_len_mul_2, _ = output_seq.size()
        denoised_seq = torch.zeros((batch_size, seqlen, individual_bits), dtype=torch.int)
        for b in range(batch_size):
            idx = 0
            for i in range(seqlen):
                for j in range(individual_bits):
                    bit_probs = output_seq[b, idx:idx + 2, :]
                    if bit_probs.numel() > 0:
                        bit = bit_probs.argmax()
                    else:
                        bit = 0
                    #bit = output_seq[b, idx:idx + 2, :].argmax()
                    denoised_seq[b, i, j] = bit
                    idx += 2
        denoised_seq = torch.where(denoised_seq == 0, torch.tensor([-1]), denoised_seq).float()
        return denoised_seq
'''