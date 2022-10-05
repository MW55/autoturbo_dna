# -*- coding: utf-8 -*-

import json

SYNTHESIS_ERROR_JSON = "config/error_sources/synthesis.json"
STORAGE_ERROR_JSON = "config/error_sources/storage.json"
SEQUENCING_ERROR_JSON = "config/error_sources/sequencing.json"


class ErrorSource(object):
    def __init__(self, process=None, config_file=None):
        """
        Class is used to load error probabilities for a given process.

        :param process: Name of the process which should be loaded.
        :param config_file: Path to the json file that contains the meta data for the error source.

        :note: By default the predefined json is used for a process, these json files can also be modified.
        It is also possible to specify a different json file, if neither the process nor a file is specified,
        an error will occur.
        """
        if config_file is None:
            if process is None:
                raise ValueError('A process must be selected or a config file must be specified!')
            elif process == "synthesis":
                config_file = SYNTHESIS_ERROR_JSON
            elif process == "storage":
                config_file = STORAGE_ERROR_JSON
            elif process == "sequencing":
                config_file = SEQUENCING_ERROR_JSON
        with open(config_file) as json_file:
            ErrorSource.config = json.load(json_file)

    config = {}

    @staticmethod
    def get_by_id(u_id, multiplier=None):
        """
        Returns the error rate and attributes of a method from a given process.

        :param u_id: The id of the method of the process in the json config file.
        :param multiplier: Can be used to increase the raw rate.
        :return: Returns a dictionaries with error rate and attributes.
        """
        config = {}
        if ErrorSource.config is None:
            return None
        if len(ErrorSource.config) >= 1:
            config = ErrorSource.config[u_id]

        if multiplier:
            config["err_rate"]['raw_rate'] = config["err_rate"]['raw_rate'] * multiplier

        return {"err_rate": config["err_rate"], "err_attributes": config["err_attributes"]}
