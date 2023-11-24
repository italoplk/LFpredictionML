import Params
import LightField

import os


class DataSet:
    def __init__(self, params: Params):
        self.path = params.word
        self.num_views_hor: params.word
        self.num_views_ver: params.word
        self.resol_ver: params.word
        self.resol_hor: params.word
        self.bit_depth = params.word
        self.list_paths = []
        self.load_paths()

    def load_paths(self):
        for lf_class in os.listdir(self.path):
            path_class = os.path.join(self.path, lf_class)
            for lf in os.listdir(path_class):

                self.list_paths.append(LightField(path_class, params))
