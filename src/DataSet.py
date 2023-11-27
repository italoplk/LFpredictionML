
from glob import iglob
from LightField import LightField
from multipledispatch import dispatch

import os
import random as rand

class DataSet:
    def __init__(self, params):
        # self.path = params.word
        # self.num_views_hor: params.word
        # self.num_views_ver: params.word
        # self.resol_ver: params.word
        # self.resol_hor: params.word
        # self.bit_depth = params.word
        self.path = params.dataset_path
        self.list_lfs = []
        self.list_train = []
        self.list_test = []
        self.test_lf_names = ["Bikes", "Danger_de_Mort", "Fountain_&_Vincent_2", "Stone_Pillars_Outside"]

        try:
            self.load_paths()
        except RuntimeError as e:
            print("Failed to load LFs: ", e)
            exit(11)

        if len(self.list_lfs) == 0:
            print("Failed to find LFs at path: ", self.path)
            exit(12)
    # TODO add new dataset variables at __str__
    def __str__(self):
        return ', '.join([self.path])

    def load_paths(self):
        for lf_path in iglob(f"{self.path}/*/*"):
            self.list_lfs.append(LightField(lf_path))


    #TODO method not finished. Finish if/when necessary
    # should split the dataset in train and test redefining the list_test acconrding to the train%
    @dispatch(float)
    def split(self, train_percentage: float):
        train_size = int(len(self.list_lfs) * train_percentage)
        test_size = len(self.list_lfs) - train_size
        for lf in len(self.list_lfs):
            if rand.random() < train_percentage:
                self.list_train.append(lf)

    @dispatch()
    def split(self):
        for lf in self.list_lfs:
            if lf.name not in self.test_lf_names:
                self.list_train.append(lf)
            else:
                self.list_test.append(lf)





