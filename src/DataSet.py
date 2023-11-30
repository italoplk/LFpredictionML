
import random as rand
from glob import iglob
from typing import List

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from LightField import LightField


# from multipledispatch import dispatch

class DataSet:
    def __init__(self, params):
        self.num_views_hor: params.num_views_hor
        self.num_views_ver: params.num_views_ver
        self.resol_ver: params.resol_ver
        self.resol_hor: params.resol_hor
        self.bit_depth = params.bit_depth
        self.path = params.dataset_path
        self.limit_train = params.limit_train
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

    # @dispatch()
    def split(self):
        for lf in self.list_lfs.inner_storage:
            if lf.name not in self.test_lf_names:
                if (len(self.list_train) < self.limit_train or self. limit_train == -1 ):
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
        X = X.load_lf()
        for transform in self.transforms:
            X = transform(X)
        return X
    def __len__(self):
        return len(self.inner_storage)

class LensletBlockedReferencer(Dataset):
    # possible TODO: make MI_size take a tuple
    def __init__(self, decoded, original, MI_size, N=32):
        super().__init__()
        self.decoded = decoded[0, :1, :, :]
        self.original = original[0, :1, :, :]
        self.N = N * MI_size
        self.inner_shape = decoded.shape
        assert(self.decoded.shape == self.original.shape)
        self.shape = tuple(dim // self.N - 1 for dim in self.inner_shape[-2:])
        assert(all(dim != 0 for dim in self.shape))
        self.len = self.shape[0] * self.shape[1]
    def __len__(self):
        return self.len
    def __getitem__(self, x):
        if x < -len(self) or x >= len(self):
            raise IndexError(x)
        elif x < 0:
            x += len(self)
        i, j = (x % self.shape[0], x // self.shape[0])
        section = self.decoded[:, i * self.N : (i+2) * self.N, j * self.N : (j+2) * self.N]
        """neighborhood = torch.ones(section.shape[0] + 1, *section.shape[1:], dtype=torch.float32)
        neighborhood[:-1, :, :, :, :] = section.to(neighborhood)
        neighborhood[:, :, :, self.N:, self.N:] = 0
        expected_block = self.original[:, :, :, i * self.N : (i+2) * self.N, j * self.N : (j+2) * self.N].to(neighborhood)"""
        neighborhood = torch.zeros(section.shape[0], *section.shape[1:], dtype=torch.float32)
        neighborhood[:, :, :] = section.to(neighborhood)
        neighborhood[:, self.N:, self.N:] = neighborhood[:, :self.N, :self.N].flip((-1,-2))
        expected_block = self.original[:, i * self.N : (i+2) * self.N, j * self.N : (j+2) * self.N].to(neighborhood)
        #print(neighborhood.shape)
        return neighborhood, expected_block
