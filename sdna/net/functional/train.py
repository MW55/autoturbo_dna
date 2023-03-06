# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as func
#testing
from sdna.net.core.channel import MarkovModelKL


def train(model, optimizer, args, epoch=1, mode="encoder"):
    """
    Trains the model depending on the mode and optimizer.

    :param model: The entire model or rather the network.
    :param optimizer: The optimizer to be used with the correctly filtered weights.
    :param args: Arguments as dictionary.
    :param epoch: In which epoch the training is right now.
    :param mode: Which section of the model is currently being trained: coder, encoder, decoder or all.
    :return: Returns the loss value of the model for the selected mode.
    """
    model.train()
    loss = 0.0

    if args['encoder'] == 'rnnatt':
        hidden = model.enc.initHidden()

    #testing
    #kl = MarkovModelKL(3)
    #kl.fit("/home/wintermute/projects/autoturbo_dna/eval/cw_40_60_hp3.fasta")
    #done

    for i in range(0, int(args["blocks"] / args["batch_size"])):
        optimizer.zero_grad()
        #print(i)
        ### testing
        #rand = np.random.randint(0, 100)
        #if rand < 101:
        #    x_train = torch.randint(0, 1, (args["batch_size"], args["block_length"], 1), dtype=torch.float)
        #elif rand < 20:
        #    print("under 20")
        #    x_train = torch.randint(1, 2, (args["batch_size"], args["block_length"], 1), dtype=torch.float)
        #test done
        #else:
        x_train = torch.randint(0, 2, (args["batch_size"], args["block_length"], 1), dtype=torch.float)
        #x_train = torch.cat((x_train, torch.zeros(256, 16, 1)), dim=1)



        padding = 0 if mode == "encoder" or mode == "decoder" else args["block_padding"] #TODO turn on padding all the time?
        #padding = args["block_padding"]
        if args['encoder'] == 'rnnatt':
            s_dec, s_enc, c_dec, noisy = model(x_train, padding=padding, seed=args["seed"] + epoch, hidden=hidden)
            hidden = s_dec[1]
            s_dec = s_dec[0]
        else:
            s_dec, s_enc, c_dec, noisy = model(x_train, padding=padding, seed=args["seed"] + epoch)
            ###
            #if i == 0:
            #    print("padding: " + str(padding))
                ###
        if mode == "all" or mode == "encoder" or mode == "decoder":
            s_dec = torch.clamp(s_dec, 0.0, 1.0)
            if args['encoder'] == 'rnnatt':
                hidden = s_dec[1]
            gradient = func.binary_cross_entropy(s_dec, x_train)
            #if mode == "encoder":   # weakens the gradients of the encoder when the generated code is unstable
            #    gradient += model.channel.evaluate(s_enc) #kl, *1.5   # TODO: find a better way to punish the net THE *1.5 is experimental!
            #    if args["encoder"] == "vae":
            #        gradient += args['beta']*((model.enc.kl_1 + model.enc.kl_2 + model.enc.kl_3)/3)
                #if np.random.randint(0, 2) == 0:
                #    gradient += model.channel.evaluate(s_enc)/2
                #else:
                #    gradient.data = torch.tensor(model.channel.evaluate(s_enc) / 2)
            #if mode == "decoder":
            #    beta = 1 # Make this a hyperparameter
             #   if args["encoder"] == "vae":
             #       gradient += beta * ((model.enc.kl_1 + model.enc.kl_2 + model.enc.kl_3)/3)
        else:
            gradient = func.mse_loss(s_enc, c_dec)
        gradient.backward()

        loss += float(gradient.item())
        optimizer.step()
    loss /= (args["blocks"] / args["batch_size"])
    return loss


def validate(model, args, epoch=1, mode="encoder", hidden=None):
    """
    Validates the model depending on the mode.

    :param model: The entire model or rather the network.
    :param args: Arguments as dictionary.
    :param epoch: In which epoch the validation is right now.
    :param mode: Which section of the model is currently being trained: coder, encoder, decoder or all.
    :return: Returns the accuracy of the model for the selected mode and the strength of the noise.
    """
    model.eval()
    accuracy = 0.0
    stability = 0.0
    noise = 0.0
    code_rate = 1
    full_corr = 0

    with torch.no_grad():
        for i in range(0, int(args["blocks"] / args["batch_size"])):
            x_val = torch.randint(0, 2, (args["batch_size"], args["block_length"], 1), dtype=torch.float)
            #x_val = torch.cat((x_val, torch.zeros(256, 16, 1)), dim=1)
            #x_val = torch.randint(0, 1, (args["batch_size"], args["block_length"], 1), dtype=torch.float)

            padding = 0 if mode == "encoder" or mode == "decoder" else args["block_padding"]
            if args['encoder'] == 'rnnatt':
                s_dec, s_enc, c_dec, noisy = model(x_val, padding=padding, seed=args["seed"] + epoch, hidden=hidden)
                hidden = s_dec[1]
                s_dec = s_dec[0]
            else:
                s_dec, s_enc, c_dec, noisy = model(x_val, padding=padding, seed=args["seed"] + epoch, validate=True)
            #print("validate")
            stability += (1.0 - model.channel.evaluate(s_enc.detach()))

            if mode == "all" or mode == "encoder" or mode == "decoder":
                s_dec = torch.clamp(s_dec, 0.0, 1.0)
                s_dec = torch.round(s_dec.detach())
                #single_acc = torch.sum(s_enc.eq(c_dec.detach())).item()
                #if i == 1:
                #    print("single accuracy: " + str(single_acc))
                accuracy += torch.sum(s_dec.eq(x_val.detach())).item()
            else:
                code_rate = s_enc.size()[2]
                s_enc = torch.round(s_enc.detach())
                single_acc = torch.sum(s_enc.eq(c_dec.detach())).item()
                if i == 1:
                    print("single accuracy: " + str(single_acc))
                accuracy += torch.sum(s_enc.eq(c_dec.detach())).item()

            equal = torch.sum(s_enc.detach().eq(noisy.detach()[:s_enc.size()[0], :s_enc.size()[1], :s_enc.size()[2]]))
            noise += (s_enc.size()[0] * s_enc.size()[1] * s_enc.size()[2]) - equal.item()
            flat_inp = x_val.detach().flatten(start_dim=1).tolist()
            flat_outp = s_dec.detach().flatten(start_dim=1).tolist()
            corr = 0
            for i in range(len(flat_inp)):
                if flat_inp[i] == flat_outp[i]:
                    corr+=1
            full_corr += corr/args["batch_size"]
            #print(corr)

    full_corr /= (args["blocks"] / args["batch_size"])
    print("Correct packages percentage: " + str(full_corr))
    accuracy /= (args["blocks"] * args["block_length"] * code_rate) #+ int(args["redundancy"])
    stability /= (args["blocks"] / args["batch_size"])
    noise /= (args["blocks"] * args["block_length"] * 3.0)
    return accuracy, stability, noise


