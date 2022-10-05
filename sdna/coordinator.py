# -*- coding: utf-8 -*-

import os
import datetime
import uuid
import json
import torch

from sdna.sim import *
from sdna.net import *
from sdna.net.functional import *
from sdna.net.functional.bitIO import BitEncodeStream, BitDecodeStream


class SDNA(object):
    def __init__(self, arguments):
        """
        Class manages the process of the program, depending on what the user passes in arguments.

        :param arguments: Arguments as dictionary.
        """
        self.args = arguments

        modes = [self.args["ids"], self.args["train"], self.args["simulate"] is not None,
                 self.args["encode"] is not None, self.args["decode"] is not None, self.args["bitenc"] is not None,
                 self.args["bitdec"] is not None]
        if sum(modes) >= 2:
            print("It is not possible to call more than one of the following arguments at the same time: --train, --encode, --decode, --simulate, --ids.")
            exit(0)

        # Set threads
        torch.set_num_threads(self.args["threads"])

        # DNA synthesis, storage, sequencing ids
        if self.args["ids"]:
            SDNA._show_error_rates()
            exit(0)

        # DNA synthesis, storage, sequencing simulator
        if self.args["simulate"]:
            code = self.args["simulate"]
            if all(b in "ACGT" for b in code):
                print("DNA synthesis, storage, sequencing result: \n(in) {}".format(code))
                r = Sim(arguments).apply_errors(code, self.args['seed'])
                print("(out) {}".format(r.replace(' ', '')))
            else:
                print("Please specify a valid code that can be modified: {}.".format(code))
            exit(0)

        load = self._check_working_directory()
        if not load:
            if self.args["encode"] or self.args["decode"]:
                print("Specify a valid model with --wdir to encode or decode!")
                exit(0)
            self.args["epoch_i"] = 0
        else:
            config = SDNA._load_config(self.args["working_dir"])
            self._check_config(config)
        SDNA._save_config(self.args["working_dir"], self.args)

        if torch.cuda.is_available() and self.args["gpu"]:
            torch.set_default_tensor_type("torch.cuda.FloatTensor")  # change the default datatype to run calculations on gpu
        self.nn = Net(self.args, load)

        # NN bits encoding
        if self.args["bitenc"]:
            code = self.args["bitenc"]
            if len(code) != self.args["block_length"]:
                print("The bit stream to be encoded must have the same length as the blocks: {}.".format(self.args["block_length"]))
                exit(0)
            if all(b in "01" for b in code):
                print("NN encoding result: \n(in) {}".format(code))
                r = encode(self.args, self.nn)
                print("(out) {}".format(r))
            else:
                print("Please specify a valid bit stream that can be encoded: {}.".format(code))
            exit(0)

        # NN code decoding
        if self.args["bitdec"]:
            code = self.args["bitdec"]
            if all(b in "ACGT" for b in code):
                print("NN decoding result: \n(in) {}".format(code))
                r = decode(self.args, self.nn)
                print("(out) {}".format(r))
            else:
                print("Please specify a valid code that can be decoded: {}.".format(code))
            exit(0)

        if self.args["encode"]:
            if not os.path.exists(self.args['inp']) or not self.args['out']:
                print("encoding needs a valid input file and a path to an output file.")
                exit(1)
            enc_stream = BitEncodeStream(inp_=self.args['inp'], outp_=self.args['out'],
                            blocksize=self.args["block_length"], index_size=self.args["index_size"])
            enc_stream.read(self.args, self.nn)
            exit(0)

        if self.args["decode"]:
            if not os.path.exists(self.args['inp']) or not self.args['out']:
                print("decoding needs a valid input file and a path to an output file.")
                exit(1)
            dec_stream = BitDecodeStream(inp_=self.args['inp'], outp_=self.args['out'],
                            blocksize=self.args["block_length"], index_size=self.args["index_size"])
            dec_stream.read(self.args, self.nn)
            exit(0)

        # if self.args["decode"]:
        #     code = self.args["decode"]
        #     if all(b in "ACGT" for b in code):
        #         print("NN decoding result: \n(in) {}".format(code))
        #         r = decode(self.args, self.nn)
        #         print("(out) {}".format(r))
        #     else:
        #         print("Please specify a valid code that can be decoded: {}.".format(code))
        #     exit(0)

        # NN training
        if self.args["train"]:
            for i, result in enumerate(self.nn.train()):
                if i == 0:
                    summary = "#Epoch #Date #Time #" + "#".join(["{} ".format(k) for k, v in result.items()])
                    print(summary)
                    if self.args["epoch_i"] == 0:
                        SDNA._log_training(self.args["working_dir"], summary)
                summary = " {:<5d}".format(self.args["epoch_i"] + 1) + datetime.datetime.now().strftime(" | %d-%m-%Y | %H:%M:%S | ") + " | ".join(["{:1.5f}".format(v) for k, v in result.items()])
                SDNA._log_training(self.args["working_dir"], summary)
                print(summary)
                self.args["epoch_i"] += 1
                SDNA._save_config(self.args["working_dir"], self.args)
            exit(0)

        # if nothing has been selected
        print("None of the following arguments would be selected: --train, --encode, --decode, --simulate, --ids.")

    def _check_working_directory(self):
        """
        Checks if a working directory should be loaded or if a new one should be created.

        :returns: Boolean whether a working directory already exists.
        """
        dir_name = self.args["working_dir"]

        if dir_name is None:
            dir_name = uuid.uuid4().hex.lower()[0:8] + "/"
            self.args["working_dir"] = dir_name
        else:
            self.args["working_dir"] = self.args["working_dir"] if self.args["working_dir"][-1] == "/" else self.args["working_dir"] + "/"

        if not os.path.exists(dir_name):
            if self.args["train"]:
                os.mkdir(dir_name)
            return False
        else:
            return True

    def _check_config(self, args):
        """
        Updates the directory for arguments, some settings are immutable for already created models.

        :param args: Arguments as dictionary.
        """
        # immutable model structure properties
        del self.args["block_length"], self.args["block_padding"], self.args["encoder"], self.args["enc_units"], \
            self.args["enc_actf"], self.args["enc_dropout"], self.args["enc_layers"], self.args["enc_kernel"], \
            self.args["enc_rnn"], self.args["decoder"], self.args["dec_units"], self.args["dec_actf"], \
            self.args["dec_dropout"], self.args["dec_layers"], self.args["dec_inputs"], self.args["dec_iterations"], \
            self.args["dec_kernel"], self.args["dec_rnn"], self.args["coder"], self.args["coder_units"], \
            self.args["coder_actf"], self.args["coder_dropout"], self.args["coder_layers"], self.args["coder_kernel"], \
            self.args["coder_rnn"], self.args["init_weights"]   # TODO: inform the user that these values cannot be overwritten (warning)
        self.args = {**args, **self.args}

    @staticmethod
    def _save_config(dir_name, args, config_name="config.json"):
        """
        Saves an config file (json).

        :param dir_name: Path to the working directory.
        :param args: Arguments as dictionary.
        :param config_name: Name of the file to save.
        """
        with open(dir_name + config_name, 'w', encoding="utf-8") as f:
            json.dump(args, f, ensure_ascii=False, indent=4)

    @staticmethod
    def _load_config(dir_name, config_name="config.json"):
        """
        Loads an existing config file (json).

        :param dir_name: Path to the working directory.
        :param config_name: Name of the file to load.
        :return: Arguments as dictionary.
        """
        with open(dir_name + config_name, 'r', encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _log_training(dir_name, msg, summary_name="summary.txt"):
        """
        Appends an message to summary file (txt).

        :param dir_name: Path to the working directory.
        :param msg: Message to be appended.
        :param summary_name: Name of the file to save.
        """
        with open(dir_name + summary_name, 'a', encoding="utf-8") as f:
            f.write(msg + "\n")

    @staticmethod
    def _show_error_rates():
        """
        Prints a list of the ids of the different default error rates.
        """
        processes = ["synthesis", "storage", "sequencing"]
        for process in processes:
            config = ErrorSource(process=process).config
            print("{} error rates:".format(process.capitalize()))
            print("\t %-4s%-28s%-28s%-12s" % ("ID", "Name", "Category", "Type"))
            for k, v in config.items():
                print("\t %-4s%-28s%-28s%-12s" % (k,
                                                  v["name"][:24] + (v["name"][24:] and "..."),
                                                  v["category"][:24] + (v["category"][24:] and "..."),
                                                  v["type"].upper()))
