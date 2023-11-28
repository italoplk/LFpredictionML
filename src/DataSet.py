
from glob import iglob
from typing import List
from LightField import LightField
from multipledispatch import dispatch

import os
import random as rand
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class DataSet:
    def __init__(self, params):
        self.num_views_hor: params.num_views_hor
        self.num_views_ver: params.num_views_ver
        self.resol_ver: params.resol_ver
        self.resol_hor: params.resol_hor
        self.bit_depth = params.bit_depth
        self.path = params.dataset_path
        self.list_lfs = LazyList([], transforms = [ToTensor()])
        self.list_train = LazyList([], transforms = [ToTensor()])
        self.list_test = LazyList([], transforms = [ToTensor()])
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


    @classmethod
    def random_split(cls, list_lfs : list, train_percentage: float):
        train_size = int(len(list_lfs) * train_percentage)
        shuffled_lfs = rand.shuffle(list_lfs)
        list_train = shuffled_lfs[:train_size]
        list_validation = shuffled_lfs[train_size:]
        return (list_train, list_validation)

    @dispatch()
    def split(self):
        for lf in self.list_lfs.inner_storage:
            if lf.name not in self.test_lf_names:
                self.list_train.append(lf)
            else:
                self.list_test.append(lf)


class LazyList(Dataset):
    def __init__(self, inner_storage : List, transforms):
        self.inner_storage = inner_storage
        self.transforms = transforms
    def append(self, elem : LightField):
        self.inner_storage.append(elem)
    def __getitem__(self, i_index):
        X = self.inner_storage[i_index]
        print(X)
        X = X.load_lf()
        for transform in self.transforms:
            X = transform(X)
        return X
    def __len__(self):
        return len(self.inner_storage)



