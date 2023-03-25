# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as func
#testing
from sdna.net.core.channel import MarkovModelKL


def train(model, optimizer, args, epoch=1, mode="encoder", warmup=False):
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
    #testing
    huber_loss = torch.nn.SmoothL1Loss(beta=1.0)
    #tesing_done

    for i in range(0, int(args["blocks"] / args["batch_size"])):
        if mode == "combined":
            for opt in optimizer:
                opt.zero_grad()
        else:
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

        if args["all_errors"] and not warmup:
            padding = args["block_padding"]
            s_dec, s_enc, c_dec, noisy = model(x_train, padding=padding, seed=args["seed"] + epoch, validate=True)
            s_dec = torch.clamp(s_dec, 0.0, 1.0)
            if args['encoder'] == 'rnnatt':
                hidden = s_dec[1]
            gradient = huber_loss(s_dec, x_train)
            gradient.backward()
            loss += float(gradient.item())

            torch.nn.utils.clip_grad_norm(model.parameters(), 1)  # 0.5

            if mode == "combined":
                for opt in optimizer:
                    opt.step()
            else:
                optimizer.step()
        else:
            padding = 0 if mode == "encoder" or mode == "decoder" else args["block_padding"] #TODO turn on padding all the time?
            #padding = args["block_padding"]
            if args['encoder'] == 'rnnatt':
                s_dec, s_enc, c_dec, noisy = model(x_train, padding=padding, seed=args["seed"] + epoch, hidden=hidden)
                hidden = s_dec[1]
                s_dec = s_dec[0]
            elif mode == 'combined':
                s_dec, s_enc, c_dec, noisy = model(x_train, padding=padding, seed=args["seed"] + epoch, validate=True)
            else:
                s_dec, s_enc, c_dec, noisy = model(x_train, padding=padding, seed=args["seed"] + epoch)
                ###
                #if i == 0:
                #    print("padding: " + str(padding))
                    ###
            if mode == "all" or mode == "encoder" or mode == "decoder" or mode == 'combined':
                s_dec = torch.clamp(s_dec, 0.0, 1.0)
                if args['encoder'] == 'rnnatt':
                    hidden = s_dec[1]
                gradient = huber_loss(s_dec, x_train)
                #gradient = func.binary_cross_entropy(s_dec, x_train)
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

            elif mode == "coder1":
                x_sys_enc = s_enc[:, :, 0].view((s_enc.size()[0], s_enc.size()[1], 1))
                x_sys_coder = c_dec[:, :, 0].view((c_dec.size()[0], c_dec.size()[1], 1))
                gradient = huber_loss(x_sys_enc, x_sys_coder)

                #get_same_packages(noisy[:, :, 0].view((noisy.size()[0], noisy.size()[1], 1)), x_sys_enc, 2, 0)
                '''
                flat_noise = torch.flatten(noisy[:, :, 0].view((noisy.size()[0], noisy.size()[1], 1)), start_dim=1)[:, :-2]
                flat_x_sys_enc = torch.flatten(x_sys_enc, start_dim=1)
    
                same_x_sys = 0
                for i in range(flat_noise.shape[0]):
                    if torch.all(flat_noise[i] == flat_x_sys_enc[i]):
                        same_x_sys += 1
                print("xsys: " + str(same_x_sys))
                '''
            elif mode == "coder2":
                x_p1_enc = s_enc[:, :, 1].view((s_enc.size()[0], s_enc.size()[1], 1))
                x_p1_coder = c_dec[:, :, 1].view((c_dec.size()[0], c_dec.size()[1], 1))
                gradient = huber_loss(x_p1_enc, x_p1_coder)
                #get_same_packages(noisy[:, :, 1].view((noisy.size()[0], noisy.size()[1], 1)), x_p1_enc, 2, 0)
                '''
                flat_noise_1 = torch.flatten(noisy[:, :, 1].view((noisy.size()[0], noisy.size()[1], 1)), start_dim=1)[:, :-2]
                flat_x_sys_enc = torch.flatten(x_p1_enc, start_dim=1)
    
                same_x_p1 = 0
                for i in range(flat_noise_1.shape[0]):
                    if torch.all(flat_noise_1[i] == flat_x_sys_enc[i]):
                        same_x_p1 += 1
                print("x_p1: " + str(same_x_p1))
                '''
            elif mode == "coder3":
                x_p2_enc = s_enc[:, :, 2].view((s_enc.size()[0], s_enc.size()[1], 1))
                x_p2_coder = c_dec[:, :, 2].view((c_dec.size()[0], c_dec.size()[1], 1))
                gradient = huber_loss(x_p2_enc, x_p2_coder)
                #get_same_packages(noisy[:, :, 2].view((noisy.size()[0], noisy.size()[1], 1)), x_p2_enc, 2, 0)
            else:
                gradient = huber_loss(s_dec, x_train)
                #gradient = huber_loss(s_enc, c_dec)
                #gradient = func.mse_loss(s_enc, c_dec)
            gradient.backward()
            #if mode in ["coder1", "coder2", "coder3"]:
            #    print(gradient.item())
            loss += float(gradient.item())

            #testing
            torch.nn.utils.clip_grad_norm(model.parameters(), 1) #0.5
            ###

            if mode == "combined":
                for opt in optimizer:
                    opt.step()
            else:
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
    full_corr_x_sys = 0
    full_corr_x_p1 = 0
    full_corr_x_p2 = 0

    with torch.no_grad():
        for i in range(0, int(args["blocks"] / args["batch_size"])):
            x_val = torch.randint(0, 2, (args["batch_size"], args["block_length"], 1), dtype=torch.float)
            #x_val = torch.cat((x_val, torch.zeros(256, 16, 1)), dim=1)
            #x_val = torch.randint(0, 1, (args["batch_size"], args["block_length"], 1), dtype=torch.float)
            '''
            padding = args["block_padding"]
            s_dec, s_enc, c_dec, noisy = model(x_val, padding=padding, seed=args["seed"] + epoch, validate=True)
            stability += (1.0 - model.channel.evaluate(s_enc.detach()))
            s_dec = torch.clamp(s_dec, 0.0, 1.0)
            s_dec = torch.round(s_dec.detach())
            accuracy += torch.sum(s_dec.eq(x_val.detach())).item()
            '''
            padding = 0 if mode == "encoder" or mode == "decoder" else args["block_padding"]
            if args['encoder'] == 'rnnatt':
                s_dec, s_enc, c_dec, noisy = model(x_val, padding=padding, seed=args["seed"] + epoch, hidden=hidden)
                hidden = s_dec[1]
                s_dec = s_dec[0]
            else:
                s_dec, s_enc, c_dec, noisy = model(x_val, padding=padding, seed=args["seed"] + epoch, validate=True)
            #print("validate")
            if not args["continuous"]:
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

            if not args["continuous"] and not args["channel"] == "basic_dna":
                equal = torch.sum(s_enc.detach().eq(noisy.detach()[:s_enc.size()[0], :s_enc.size()[1], :s_enc.size()[2]]))
                noise += (s_enc.size()[0] * s_enc.size()[1] * s_enc.size()[2]) - equal.item()
            else:
                noise = 1
            flat_inp = x_val.detach().flatten(start_dim=1).tolist()
            flat_outp = s_dec.detach().flatten(start_dim=1).tolist()
            corr = 0
            for i in range(len(flat_inp)):
                if flat_inp[i] == flat_outp[i]:
                    corr+=1
            full_corr += corr/args["batch_size"]
            #print(corr)
            corr_x_sys, corr_x_p1, corr_x_p2 = get_correct_coder(s_enc, c_dec, args)
            full_corr_x_sys += corr_x_sys
            full_corr_x_p1 += corr_x_p1
            full_corr_x_p2 += corr_x_p2


    full_corr /= (args["blocks"] / args["batch_size"])
    full_corr_x_sys /= (args["blocks"] / args["batch_size"])
    full_corr_x_p1 /= (args["blocks"] / args["batch_size"])
    full_corr_x_p2 /= (args["blocks"] / args["batch_size"])
    print("Correct packages percentage: " + str(full_corr))
    print("Coder correct reconstruction, x_sys: " + str(full_corr_x_sys) + " x_p1: " + str(full_corr_x_p1)
          + " x_p2: " + str(full_corr_x_p2))
    accuracy /= (args["blocks"] * args["block_length"] * code_rate) #+ int(args["redundancy"])
    stability /= (args["blocks"] / args["batch_size"])
    noise /= (args["blocks"] * args["block_length"] * 3.0)
    return accuracy, stability, noise


def get_correct_coder(s_enc, c_dec, args):
    x_sys_enc = s_enc[:, :, 0].view((s_enc.size()[0], s_enc.size()[1], 1))
    x_sys_enc = x_sys_enc.detach().flatten(start_dim=1).tolist()
    x_p1_enc = s_enc[:, :, 1].view((s_enc.size()[0], s_enc.size()[1], 1))
    x_p1_enc = x_p1_enc.detach().flatten(start_dim=1).tolist()
    x_p2_enc = s_enc[:, :, 2].view((s_enc.size()[0], s_enc.size()[1], 1))
    x_p2_enc = x_p2_enc.detach().flatten(start_dim=1).tolist()

    x_sys_dec = c_dec[:, :, 0].view((c_dec.size()[0], c_dec.size()[1], 1))
    x_sys_dec = x_sys_dec.detach().flatten(start_dim=1).tolist()
    x_p1_dec = c_dec[:, :, 1].view((c_dec.size()[0], c_dec.size()[1], 1))
    x_p1_dec = x_p1_dec.detach().flatten(start_dim=1).tolist()
    x_p2_dec = c_dec[:, :, 2].view((c_dec.size()[0], c_dec.size()[1], 1))
    x_p2_dec = x_p2_dec.detach().flatten(start_dim=1).tolist()

    x_sys_corr = 0
    x_p1_corr = 0
    x_p2_corr = 0
    full_corr_x_sys = 0
    full_corr_x_p1 = 0
    full_corr_x_p2 = 0

    for i in range(len(x_sys_enc)):
        if x_sys_enc[i] == x_sys_dec[i]:
            x_sys_corr += 1
        if x_p1_enc[i] == x_p1_dec[i]:
            x_p1_corr += 1
        if x_p2_enc[i] == x_p2_dec[i]:
            x_p2_corr += 1
    full_corr_x_sys += x_sys_corr / args["batch_size"]
    full_corr_x_p1 += x_p1_corr / args["batch_size"]
    full_corr_x_p2 += x_p2_corr / args["batch_size"]

    return full_corr_x_sys, full_corr_x_p1, full_corr_x_p2


def get_same_packages(x, baseline, x_padding=None, baseline_padding=None):
    if x_padding:
        flat_x = torch.flatten(x, start_dim=1)[:, :-x_padding]
    else:
        flat_x = torch.flatten(x, start_dim=1)
    if baseline_padding:
        flat_baseline = torch.flatten(baseline, start_dim=1)[:, :-baseline_padding]
    else:
        flat_baseline = torch.flatten(baseline, start_dim=1)

    same_x = 0
    for i in range(flat_x.shape[0]):
        if torch.all(flat_x[i] == flat_baseline[i]):
            same_x += 1
    print(str(same_x))
