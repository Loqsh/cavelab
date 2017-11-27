import json

class hparams(dict):
    __getattr__= dict.__getitem__
    __setattr__= dict.__setitem__
    __delattr__= dict.__delitem__
    def __init__(self, path="hparams.json", name="default"):
        self.hparams = self.load_hparams(path, name)
        self.wrap_body(self.hparams)

    def load_hparams(self, path, name):
        with open(path) as data_file:
            data = json.load(data_file)
        return data[name]

    def wrap_body(self, hparams):
        for k, v in hparams.items():
            self[k]=v["data"]
