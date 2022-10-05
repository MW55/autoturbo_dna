# -*- coding: utf-8 -*-

import torch

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
        self.model = AutoEncoder(self.args,
                                 ENCODERS[self.args["encoder"]](self.args),
                                 DECODERS[self.args["decoder"]](self.args),
                                 CODERS[self.args["coder"]](self.args),
                                 Channel(self.args))

        if torch.cuda.device_count() > 1 and self.args["gpu"] and self.args["parallel"]:
            self.model.enc.set_parallel()
            self.model.dec.set_parallel()
            self.model.coder.set_parallel()

        if not load:
            wf = func.initialize(method=self.args["init_weights"])
            if wf is not None:
                self.model.apply(wf)     # initialize weights and biases
        else:
            self.model = Net._load_model(self.args["working_dir"], self.model)
        Net._save_model(self.args["working_dir"], self.model)

    def _initialize_optimizers(self):
        """
        Returns the individual optimizers as tuples in the following order: autoencoder, encoder, decoder, coder.

        :returns: The individual optimizers in a tuple.
        """
        if not self.args["simultaneously_training"]:        # create separate optimizer for encoder and decoder or one for both
            enc_optimizer = Net.optimizers(self.args["enc_optimizer"])(
                filter(lambda p: p.requires_grad, self.model.enc.parameters()), lr=self.args["enc_lr"])
            dec_optimizer = Net.optimizers(self.args["dec_optimizer"])(
                filter(lambda p: p.requires_grad, self.model.dec.parameters()), lr=self.args["dec_lr"])
            ae_optimizer = None
        else:
            ae_optimizer = Net.optimizers(self.args["enc_optimizer"])(
                filter(lambda p: p.requires_grad, list(self.model.enc.parameters()) + list(self.model.dec.parameters())), lr=self.args["enc_lr"])
            enc_optimizer = None
            dec_optimizer = None

        coder_optimizer = Net.optimizers(self.args["coder_optimizer"])(
            filter(lambda p: p.requires_grad, self.model.coder.parameters()), lr=self.args["coder_lr"])
        return ae_optimizer, enc_optimizer, dec_optimizer, coder_optimizer

    def train(self):
        """
        Trains the network as specified.

        :returns: The results of the runs are returned as a generator.
        """
        ae_optimizer, enc_optimizer, dec_optimizer, coder_optimizer = self._initialize_optimizers()
        for epoch in range(1, self.args["epochs"] + 1):
            res = dict()
            if self.args["simultaneously_training"]:    # train modules of model simultaneously or separately from each other
                for i in range(self.args["enc_steps"]):
                    res["Encoder"] = res["S-Decoder"] = func.train(self.model, ae_optimizer, self.args, epoch=epoch, mode="encoder")
                for i in range(self.args["coder_steps"]):
                    res["I-Decoder"] = func.train(self.model, coder_optimizer, self.args, epoch=epoch, mode="coder")
                res["Accuracy"], res["Stability"], res["Noise"] = func.validate(self.model, self.args, epoch=epoch, mode="all")
            else:
                for i in range(self.args["enc_steps"]):
                    res["Encoder"] = func.train(self.model, enc_optimizer, self.args, epoch=epoch, mode="encoder")
                for i in range(self.args["dec_steps"]):
                    res["S-Decoder"] = func.train(self.model, dec_optimizer, self.args, epoch=epoch, mode="decoder")
                for i in range(self.args["coder_steps"]):
                    res["I-Decoder"] = func.train(self.model, coder_optimizer, self.args, epoch=epoch, mode="coder")
                res["Accuracy"], res["Stability"], res["Noise"] = func.validate(self.model, self.args, epoch=epoch, mode="all")

            Net._save_model(self.args["working_dir"], self.model)
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
