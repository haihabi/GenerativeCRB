import argparse
import os
import json
import copy


class ConfigReader(object):
    def __init__(self):
        self.arg_dict = dict()
        self.enum_dict = dict()
        self.parameters = None

    def add_parameter(self, name, **kwargs):
        if kwargs.get('enum'):
            self.enum_dict[name] = kwargs.get('enum')
            kwargs.pop('enum')
        self.arg_dict.update({name: kwargs})

    def _handle_enums(self, input_dict):
        for name, c in self.enum_dict.items():
            input_dict[name] = c[input_dict[name]]
        return input_dict

    def get_user_arguments(self):
        if self.parameters is None:
            argparser = argparse.ArgumentParser()
            for k, v in self.arg_dict.items():
                argparser.add_argument('--' + k, **v)
            self.parameters = argparser.parse_args()
        return self.parameters

    def read_parameters(self):
        args = copy.deepcopy(self.get_user_arguments())
        self._handle_enums(args.__dict__)
        return args

    def save_config(self, folder):
        args = self.get_user_arguments()
        with open(os.path.join(folder, 'run.config.json'), 'w') as outfile:
            json.dump(args.__dict__, outfile)

    def load_config(self, folder):
        with open(os.path.join(folder, 'run.config.json'), 'r') as outfile:
            cfg = json.load(outfile)
        self._handle_enums(cfg)
        return cfg
