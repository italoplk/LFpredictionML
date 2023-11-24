from Params import Params
from LightField import LightField

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
            for lf_name in os.listdir(path_class):
                lf_temp = LightField(path_class, lf_name)
                self.list_paths.append(lf_temp)
                print(lf_temp.__str__())


params = Params("/home/idm/Downloads/EPFLOriginal/")
DataSet(params)