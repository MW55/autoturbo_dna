# -*- coding: utf-8 -*-

import torch
from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup
import sdna.net.functional as func
from sdna.net.core import *


class Net(object):
    def __init__(self, arguments, load):
        """
        Class is used to control the models.

        :param load: Whether to load the model from a previous run.
        :param arguments: Arguments as dictionary.
        """
        self.args = arguments
        self._initialize_models(load)

    def _initialize_models(self, load=False):
        """
        Creates the individual networks as well as the model and initializes it.

        :param load: Whether to load the model from a previous run.
        """
        if not self.args['separate_coder_training']:
            self.model = AutoEncoder(self.args,
                                     ENCODERS[self.args["encoder"]](self.args),
                                     DECODERS[self.args["decoder"]](self.args),
                                     CODERS[self.args["coder"]](self.args),
                                     Channel(self.args))
        else:
            self.model = AutoEncoder(self.args,
                                     ENCODERS[self.args["encoder"]](self.args),
                                     DECODERS[self.args["decoder"]](self.args),
                                     CODERS[self.args["coder"]](self.args),
                                     Channel(self.args),
                                     CODERS[self.args["coder"]](self.args),
                                     CODERS[self.args["coder"]](self.args))

        if torch.cuda.device_count() > 1 and self.args["gpu"] and self.args["parallel"]:
            self.model.enc.set_parallel()
            self.model.dec.set_parallel()
            self.model.coder.set_parallel()
            self.model.coder2.set_parallel()
            self.model.coder3.set_parallel()

        if not load:
            wf = func.initialize(method=self.args["init_weights"])
            if wf is not None:
                self.model.apply(wf)  # initialize weights and biases
        else:
            self.model = Net._load_model(self.args["working_dir"], self.model)
        Net._save_model(self.args["working_dir"], self.model)

    def _initialize_optimizers(self):
        """
        Returns the individual optimizers as tuples in the following order: autoencoder, encoder, decoder, coder.

        :returns: The individual optimizers in a tuple.
        """
        if not self.args[
            "simultaneously_training"]:  # create separate optimizer for encoder and decoder or one for both
            enc_optimizer = Net.optimizers(self.args["enc_optimizer"])(
                filter(lambda p: p.requires_grad, self.model.enc.parameters()), lr=self.args["enc_lr"])
            dec_optimizer = Net.optimizers(self.args["dec_optimizer"])(
                filter(lambda p: p.requires_grad, self.model.dec.parameters()), lr=self.args["dec_lr"])
            ae_optimizer = None
        else:
            ae_optimizer = Net.optimizers(self.args["enc_optimizer"])(
                filter(lambda p: p.requires_grad,
                       list(self.model.enc.parameters()) + list(self.model.dec.parameters())), lr=self.args["enc_lr"])
            enc_optimizer = None
            dec_optimizer = None
        if not self.args["separate_coder_training"]:
            coder_optimizer = Net.optimizers(self.args["coder_optimizer"])(
            filter(lambda p: p.requires_grad, self.model.coder.parameters()), lr=self.args["coder_lr"])
        else:
            coder_optimizer = Net.optimizers(self.args["coder_optimizer"])(
            filter(lambda p: p.requires_grad, self.model.coder.parameters()), lr=self.args["coder_lr"]) #lr=self.args["coder_lr"]
            coder_optimizer2 = Net.optimizers(self.args["coder_optimizer"])(
            filter(lambda p: p.requires_grad, self.model.coder2.parameters()), lr=self.args["coder_lr"]) #lr=self.args["coder_lr"]
            coder_optimizer3 = Net.optimizers(self.args["coder_optimizer"])(
            filter(lambda p: p.requires_grad, self.model.coder3.parameters()), lr=self.args["coder_lr"]) #lr=self.args["coder_lr"]
        if self.args["decoder"] == "transformer": #or self.args["decoder"] == "entransformer":
            self.scheduler_lr_dec = torch.optim.lr_scheduler.ExponentialLR(dec_optimizer, gamma=0.9)
            self.scheduler_dec = create_lr_scheduler_with_warmup(self.scheduler_lr_dec,
                                                        warmup_start_value=0.0,
                                                        warmup_end_value=0.1,
                                                        warmup_duration=10)
        if self.args["encoder"] == "transformer":
            self.scheduler_lr_enc = torch.optim.lr_scheduler.ExponentialLR(enc_optimizer, gamma=0.9)
            self.scheduler_enc = create_lr_scheduler_with_warmup(self.scheduler_lr_enc,
                                                        warmup_start_value=0.0,
                                                        warmup_end_value=0.1,
                                                        warmup_duration=10)
        if not self.args["separate_coder_training"]:
            return ae_optimizer, enc_optimizer, dec_optimizer, coder_optimizer
        else:
            return ae_optimizer, enc_optimizer, dec_optimizer, coder_optimizer, coder_optimizer2, coder_optimizer3

    def train(self):
        """
        Trains the network as specified.

        :returns: The results of the runs are returned as a generator.
        """
        if not self.args['separate_coder_training']:
            ae_optimizer, enc_optimizer, dec_optimizer, coder_optimizer = self._initialize_optimizers()
        else:
            ae_optimizer, enc_optimizer, dec_optimizer, coder_optimizer, coder_optimizer2, coder_optimizer3 = self._initialize_optimizers()
        mult = self.args["amplifier"]
        last_10_acc = []
        indel_mult = 1
        self.model.channel._dna_simulator.indel_multiplier = indel_mult
        for epoch in range(1, self.args["epochs"] + 1):
            ##testing###
            if epoch == 1:
            #if epoch % 100 == 0 and mult >= 1:
            #    mult -= 1
            #    self.model.channel._dna_simulator._prepare_error_simulation(mult)
                 print("Error rate: " + str(sum([self.model.channel._dna_simulator.error_rates[i]["err_rate"]["raw_rate"]
                                                for i in range(len(self.model.channel._dna_simulator.error_rates))])))
            res = dict()
            #
            #if epoch % 10 == 0 and indel_mult > 1:
            #    indel_mult -= 1
            #    self.model.channel._dna_simulator.indel_multiplier = indel_mult
             #
            if epoch < self.args["warmup"]:
                warmup = True
            else:
                warmup = False
            if self.args[
                "simultaneously_training"]:  # train modules of model simultaneously or separately from each other
                for i in range(self.args["enc_steps"]):
                    res["Encoder"] = res["S-Decoder"] = func.train(self.model, ae_optimizer, self.args, epoch=epoch,
                                                                   mode="encoder", warmup = warmup)
                for i in range(self.args["coder_steps"]):
                    res["I-Decoder"] = func.train(self.model, coder_optimizer, self.args, epoch=epoch, mode="coder", warmup = warmup)
                res["Accuracy"], res["Stability"], res["Noise"] = func.validate(self.model, self.args, epoch=epoch,
                                                                                mode="all")
            else:
                for i in range(self.args["enc_steps"]):
                    res["Encoder"] = func.train(self.model, enc_optimizer, self.args, epoch=epoch, mode="encoder", warmup = warmup)
                for i in range(self.args["dec_steps"]):
                    res["S-Decoder"] = func.train(self.model, dec_optimizer, self.args, epoch=epoch, mode="decoder", warmup = warmup)

                for i in range(self.args["coder_steps"]):
                    if not self.args['separate_coder_training']:
                        res["I-Decoder"] = func.train(self.model, coder_optimizer, self.args, epoch=epoch, mode="coder", warmup = warmup)
                    else:
                        res["I-Decoder1"] = func.train(self.model, coder_optimizer, self.args, epoch=epoch,
                                                       mode="coder1", warmup = warmup)
                        res["I-Decoder2"] = func.train(self.model, coder_optimizer2, self.args, epoch=epoch,
                                                       mode="coder2", warmup = warmup)
                        res["I-Decoder3"] = func.train(self.model, coder_optimizer3, self.args, epoch=epoch,
                                                       mode="coder3", warmup = warmup)
                all_optimizers = [enc_optimizer, dec_optimizer,
                                  coder_optimizer] if not self.args['separate_coder_training'] else [enc_optimizer, dec_optimizer, coder_optimizer, coder_optimizer2, coder_optimizer3]
                for i in range(self.args["combined_steps"]): #combined_steps
                    res["Combined"] = func.train(self.model, all_optimizers, self.args, epoch=epoch, mode="combined", warmup = warmup)

                res["Accuracy"], res["Stability"], res["Noise"], res["CorrectBlocks"] = func.validate(self.model, self.args, epoch=epoch,
                                                                                mode="all")
            '''
            if not self.args["batch_size"] >= 1024:
                last_10_acc.append(res["Accuracy"])
                if len(last_10_acc) >= 10:
                    if max(last_10_acc) - min(last_10_acc) < 0.01:
                        self.args["batch_size"] = self.args["batch_size"] * 2
                        self.args["blocks"] = self.args["blocks"] * 2
                        print("increased batch size, batch size is now: " + str(self.args["batch_size"]))
                    del last_10_acc[0]
            '''

            '''
            if epoch % 100 == 0 and not self.args["batch_size"] >= 1024:
                self.args["batch_size"] = self.args["batch_size"]*2
                print("increased batch size, batch size is now: " + str(self.args["batch_size"]))
            '''


            last_10_acc.append(res["Accuracy"])
            if len(last_10_acc) >= 10:
                if not self.args["batch_size"] >= 1024:
                    if max(last_10_acc) - min(last_10_acc) < 0.01:
                        self.args["batch_size"] = self.args["batch_size"] * 2
                        self.args["blocks"] = self.args["blocks"] * 2
                        print("increased batch size, batch size is now: " + str(self.args["batch_size"]))
                elif self.args["enc-lr"] > 0.00005:
                    if max(last_10_acc) - min(last_10_acc) < 0.01:
                        self.args["coder_lr"] = self.args["coder_lr"] - 0.00001
                        self.args["enc_lr"] = self.args["enc_lr"] - 0.00001
                        self.args["dec_lr"] = self.args["dec_lr"] - 0.00001
                del last_10_acc[0]

            Net._save_model(self.args["working_dir"], self.model)
            if self.args["decoder"] == "transformer": #or self.args["decoder"] == "entransformer":
                #self.scheduler_lr.step()
                self.scheduler_dec(None)
                print("learning_rate_dec: " + str(dec_optimizer.param_groups[0]['lr']))
                #print("learning rate:" + str(self.scheduler_lr.get_last_lr()[0]))
            if self.args["encoder"] == "tansformer":
                self.scheduler_enc(None)
                print("learning_rate_dec: " + str(enc_optimizer.param_groups[0]['lr']))
            yield res

    @staticmethod
    def optimizers(name: str):
        """
        Helper function, returns an optimizer.

        :param name: Name of optimizer.
        :return: Returns class of the desired optimizer.
        """
        if name.lower() == "adam":
            return torch.optim.Adam
        elif name.lower() == "sgd":
            return torch.optim.SGD
        elif name.lower() == "adagrad":
            return torch.optim.Adagrad
        else:
            return torch.optim.Adam

    @staticmethod
    def _save_model(dir_name, model, model_name="model.pth"):
        """
        Saves an model.

        :param dir_name: Path to the working directory.
        :param model: The model to be saved.
        :param model_name: Name of the file to save.
        """
        torch.save(model.state_dict(), dir_name + model_name)

    @staticmethod
    def _load_model(dir_name, model, model_name="model.pth"):
        """
        Loads an existing model.

        :param dir_name: Path to the working directory.
        :param model: The model to be loaded.
        :param model_name: Name of the file to load.
        :return: The model that has been loaded with the weights.
        """
        model.load_state_dict(torch.load(dir_name + model_name))
        model.eval()
        return model
