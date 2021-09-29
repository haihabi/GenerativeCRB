import argparse
from argparse import Namespace
import os
import json
import copy

CONFIG = "config_file"


class ConfigReader(object):
    def __init__(self):
        self.arg_dict = dict()
        self.enum_dict = dict()
        self.parameters = None

    def add_parameter(self, name, **kwargs):
        if name == CONFIG:
            raise Exception(f"Cant user the argument named:{CONFIG}")
        if kwargs.get('enum'):
            self.enum_dict[name] = kwargs.get('enum')
            kwargs.pop('enum')
        self.arg_dict.update({name: kwargs})

    def _handle_enums(self, input_dict):
        for name, c in self.enum_dict.items():
            input_dict[name] = c[input_dict[name]]
        return input_dict

    def _handle_boolean(self, input_dict):
        # output_dict = copy.copy(input_dict)
        for name, c in input_dict.items():
            if isinstance(c, str):
                if c.lower() == "true":
                    input_dict[name] = True
                if c.lower() == "false":
                    input_dict[name] = False
        # return output_dict

    def _handle_enums2str(self, input_dict):
        for name, c in self.enum_dict.items():
            input_dict[name] = input_dict[name].name
        return input_dict

    def get_user_arguments(self):
        lcfg = self.load_config()  # Load Config from file
        argparser = argparse.ArgumentParser()
        for k, v in self.arg_dict.items():
            argparser.add_argument('--' + k, **v)
        parameters, _ = argparser.parse_known_args()
        parameters_dict = vars(parameters)
        for pname, pvalue in self.arg_dict.items():
            if parameters.__getattribute__(pname) == pvalue and len(lcfg > 0):  # Same as defulat
                if lcfg.get(pname) is not None:
                    parameters_dict[pname] = lcfg.get(pname)
        self._handle_enums(parameters_dict)
        self._handle_boolean(parameters_dict)
        self.parameters = Namespace(**parameters_dict)
        return self.parameters

    def read_parameters(self):
        args = copy.deepcopy(self.get_user_arguments())
        self._handle_enums(args.__dict__)
        self._handle_boolean(args.__dict__)
        return args

    def save_config(self, folder):
        args = self.get_user_arguments()
        args_dict = vars(args)
        args_dict = self._handle_enums2str(args_dict)
        with open(os.path.join(folder, 'run.config.json'), 'w') as outfile:
            json.dump(args_dict, outfile)

    def load_config(self):
        argparser = argparse.ArgumentParser()
        argparser.add_argument('--' + CONFIG, type=str, required=False)
        config_args, _ = argparser.parse_known_args()
        config_file = config_args.__getattribute__(CONFIG)
        if config_args.__getattribute__(CONFIG) is None:
            return {}
        with open(config_file, 'r') as outfile:
            cfg = json.load(outfile)
        self._handle_enums(cfg)
        self._handle_boolean(cfg)
        return cfg
