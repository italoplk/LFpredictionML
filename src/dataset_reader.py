



from functools import partial
from functools import reduce as fc_reduce
import math
from operator import __mul__
from typing import Callable, Sequence

from torch.utils.data import Dataset




#import h5py
from glob import iglob
import numpy as np
import torch
from torch.utils.data import DataLoader
from random import shuffle, sample
import torch
from itertools import accumulate
import torch.nn.functional as F

import cv2
# import einops
from einops import rearrange, reduce
from einops import EinopsError
import os
import sys
from dotenv import load_dotenv
from abc import ABC,abstractclassmethod

class chain:
    def __init__(self, *components):
        self.components = components
        self.lens = tuple(map(len, components))
        self.len = sum(self.lens)
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        if index < - len(self) or index >= len(self):
            raise IndexError(index)
        elif index < 0:
            index += self.len
        for (component, till_now, now) in zip(self.components, accumulate(self.lens, initial=0), self.lens):
            if index < till_now + now:
                return component[index - till_now]
        raise Exception("This should have been unreachable")

class mymap:
    def __init__(self, f, *components):
        self.f = f
        self.components = components
        self.len = min(map(len, components))
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        if index < -self.len or index >= self.len:
            raise IndexError(index)
        elif index < 0:
            index += self.len
        return self.f(*(comp[index] for comp in self.components))

class single:
    def __new__(cls, f, args = ()):
        return mymap(lambda ignore_me: partial(f, *args)(), [0])
    def __len__(self):
        return 1
    def __getitem__(self, i):
        if i != 0 and i != -1:
            raise IndexError(i)
        else:
            return self.f()

def pad_to_multiple(array : torch.Tensor, dims : Sequence[int]):
    shape = array.shape
    new_shape = tuple(math.ceil(dim/multiple)*multiple for dim, multiple in zip(shape, dims))
    print(new_shape)
    padding = [0]*(2*len(shape))
    padding[1:1+len(new_shape)*2:2] = (new - old for new, old in zip(new_shape, shape))
    print(padding)
    return F.pad(array, padding, 'constant', 0)

#TODO CHECAR O -1 DUPLO
normalizer_factor = 2/(2 ** 16 - 1)
def read_LF_PNG(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        # Maybe find a better exception type
        raise OSError(f'Failed to read "{str(path)}"')
    return img

# def normalize_16bit_image(image):
#     return torch.tensor(image.astype(np.float32)) * normalizer_factor - 1

def normalize_16bit_image(image):
    return torch.tensor(image.astype(np.float32)) * normalizer_factor - 1

def normalize_16bit_image_rgb(image):
    return torch.tensor(image.astype(np.float32)) / 255.0


# def unnormalize_to_16bit_image(image):
#     return (((image +1)/normalizer_factor).astype(np.uint16))

def unnormalize_to_16bit_image(image):
    return (((image+1)/normalizer_factor).astype(np.uint16))

def unnormalize_to_16bit_image_rgb(image):
    return (((image)*255.0).astype(np.uint16))

#write the LFs after validation
def write_LF_PMG(image, path):
    image = rearrange(image, 'c s t u v -> (s u) (t v) c', s=13, t=13)
    image = unnormalize_to_16bit_image(image)
    #image = cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)
    #print(image.shape)
    #print(path)
    cv2.imwrite(f'{path}', image)

def write_LF_PNG_lenslet(image, path):
    image = unnormalize_to_16bit_image(image)
    image = rearrange(image, 'c h w ->  w h c')
    #image = cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)
    #print(image.shape)
    #print(path)
    cv2.imwrite(f'{path}', image)

def read_LF(path : str, **kwargs : int) -> torch.Tensor:
    """
        path : path to the image to be read
        s, t, u, v : optional argumets... s and t assumed to be 13 if nothing is supplied;
            if supplied, then at least one in each {s, u}, {t, v} needs to be supplied...
            the image dimensions *must* be a multiple of them;
    """
    dims = kwargs if len(kwargs) != 0 else { 's' : 13, 't' : 13}
    img = read_LF_PNG(path)
    img_ycbcr = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB))
    img_normalized = normalize_16bit_image(img_ycbcr)
    img_normalized = pad_to_multiple(img_normalized, (dims['s'], dims['t']))
    return rearrange(img_normalized, '(s u) (t v) c -> c s t u v', **dims)[:1, :, :, :, :]

def read_LF_lenslet(path : str) -> torch.Tensor:
    """
        path : path to the image to be read
        Will return a 2d tensor... NOT A 4D ONE!
    """
    img = read_LF_PNG(path)
    img_ycbcr = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB))
    img_normalized_y = normalize_16bit_image(img_ycbcr)[:, :, :1]
    img_color_in_front = rearrange(img_normalized_y, '... c -> c ...')
    return img_color_in_front

on_the_list = lambda candidate, l: any(test in candidate for test in l)

class LF_pair_lister(ABC):
    @abstractclassmethod
    def read_pair(self, lfclass : str, lf : str) -> Sequence[torch.Tensor, tuple[str, torch.Tensor]]:
        raise NotImplemented()

#modificar pairwise para retornar uma lista de blocos
class pairwise_lister(LF_pair_lister):
    def __init__(self, originals, decoded, exin_clude, exclude = True, read_from_LF : Callable[[str], torch.Tensor] = read_LF, params : dict = dict()):
        self.decoded = decoded
        self.originals = originals
        if exclude:
            included = lambda candidate: not(on_the_list(candidate, exin_clude))
        else:
            included = lambda candidate: on_the_list(candidate, exin_clude)
        
        self.read_from_LF = read_from_LF
        self.original_paths = tuple(filter(included, iglob(f"{originals}/*/*.png")))
        self.decoded_paths = tuple(filter(included, iglob(f"{decoded}/*/*/*.png")))
        #print('Original', self.original_paths)

        self.original_lfs = { tuple(path.split("/")[-2:]) : path for path in self.original_paths}
        #print(self.original_paths)
        self.decoded_lfs = { tuple(path.split("/")[-3:]) : path for path in self.decoded_paths}
        self.errs = [tuple(key[:2]) for key in self.decoded_lfs if (key[0], key[1] + '.png') not in self.original_lfs]
        #print(self.errs)
        self.pairs = {
            key : (self.original_lfs[(key[0], key[1] + '.mat.png')], dec_path) for key, dec_path in self.decoded_lfs.items()}
        self.lfs = list(set(map(lambda key: key[:2], self.pairs.keys())))
        self.bbps = { lf : [bpp for lfclass, lfname, bpp in self.decoded_lfs.keys() if (lfclass, lfname) == lf] for lf in self.lfs }
        self.s = params['views_h']
        self.t = params['views_w']

    def read_original_mat(self, lfclass, lf):
        #path = '/'.join((self.originals, lfclass, lf.replace("___", "_&_") + '.mat.png'))
        path = '/'.join((self.originals, lfclass, lf + '.mat.png'))
        return self.read_from_LF(path, s=self.s, t=self.t)
        #with h5py.File(path, 'r') as f:
        #    return np.array(f['dataset'])
    
    def read_decoded_mat(self, lfclass, lf, bpp):
        path = '/'.join((self.decoded, lfclass, lf, bpp))
        return (bpp, self.read_from_LF(path, s=self.s, t=self.t))
    
    def read_pair(self, lfclass, lf):
        try:
            bpps = self.bbps[lfclass, lf]
        except KeyError:
            lf = lf.replace("___", "_&_")
            bpps = self.bbps[lfclass, lf]
        bpps = list(bpps)
        shuffle(bpps)
        original = single(self.read_original_mat, (lfclass, lf))
        #print(len(original))
        decoded = partial(self.read_decoded_mat, lfclass, lf)
        return (original, ((bpp, decoded(bpp)) for bpp in bpps))

class self_pairer:
    def __init__(self, originals : str, read_from_LF : Callable[[str], torch.Tensor] = read_LF, params : dict[str, int] = dict()):
        self.originals = originals
        self.original_paths = tuple(iglob(f"{originals}/*/*.png"))
        self.read_from_LF = read_from_LF
        keys = (path.split('/')[-2:] for path in self.original_paths)
        self.lf_by_class_and_name = { (key[0], key[1].split('.')[0]) : path for key, path in zip(keys, self.original_paths) }
        self.s = params.get('views_h', 13)
        self.t = params.get('views_w', 13)
        #print(list(self.lf_by_class_and_name.keys()))
    def read_pair(self, lfclass, lf) -> tuple[torch.Tensor, Sequence[tuple[str, torch.Tensor]]]:
        path = self.lf_by_class_and_name[(lfclass, lf)]
        data = self.read_from_LF(path, s=self.s, t=self.t)
        return (data, [('no bpp', data)]) # tensor is not copied



class fold_dataset:
    def __init__(self, lister, elems):
        self.elems = list(elems)
        self.lister = lister
    def __len__(self):
        return len(self.elems)
    def __getitem__(self, n):
        index = self.elems[n]
        return (index, self.lister(*index))
    def sampled(self, n):
        return (self[i] for i in sample(range(len(self)), n))

load_dotenv("_dl4_LL.env")

ORIGINAL_LFS_PATH_9views = os.environ["ORIGINAL_LFS_MV_RGB_9views"]
#DECODED_LFS_PATH =  os.environ["DECODED_LFS_PATH"]

training_dataset = self_pairer(ORIGINAL_LFS_PATH_9views, read_from_LF=read_LF, params = {'views_w': 9, 'views_h' : 9})
#training_dataset = pairwise_lister(ORIGINAL_LFS_PATH, DECODED_LFS_PATH, ["Bikes", "Danger_de_Mort", "Fountain___Vincent_2", "Stone_Pillars_Outside"], exclude=True)
# test_dataset = pairwise_lister(ORIGINAL_LFS_PATH, DECODED_LFS_PATH, ["Bikes", "Danger_de_Mort", "Fountain___Vincent_2", "Stone_Pillars_Outside"], exclude=False)
# test_dataset = self_pairer(ORIGINAL_LFS_PATH)


#display(list((k, v) for k,v in training_dataset.bbps.items() if len(v) != 5)[:5])


# In[3]:


#torch.tensor([0,1,2,3], dtype=torch.float32)


# In[4]:



class blocked_referencer(Dataset):
    def __init__(self, decoded, original):
        super().__init__()
        self.decoded = decoded[0, :1, :, :, :, :]
        self.original = original[0, :1, :, :, :, :]
        self.N = 32
        self.inner_shape = decoded.shape
        if(decoded.shape == original.shape):
            self.shape = tuple(dim // self.N - 1 for dim in self.inner_shape[-2:])
            assert(all(dim != 0 for dim in self.shape))
        else:
            self.shape = (0,0)
        self.len = fc_reduce(__mul__, self.shape, 1)
    def __len__(self):
        return self.len
    def __getitem__(self, x):
        if x < -len(self) or x >= len(self):
            raise IndexError(x)
        elif x < 0:
            x += len(self) 
        i, j = (x % self.shape[0], x // self.shape[0])
        section = self.decoded[:, :, :, i * self.N : (i+2) * self.N, j * self.N : (j+2) * self.N]
        """neighborhood = torch.ones(section.shape[0] + 1, *section.shape[1:], dtype=torch.float32)
        neighborhood[:-1, :, :, :, :] = section.to(neighborhood)
        neighborhood[:, :, :, self.N:, self.N:] = 0
        expected_block = self.original[:, :, :, i * self.N : (i+2) * self.N, j * self.N : (j+2) * self.N].to(neighborhood)"""
        neighborhood = torch.zeros(section.shape[0], *section.shape[1:], dtype=torch.float32)
        neighborhood[:, :, :, :, :] = section.to(neighborhood)
        neighborhood[:, :, :, self.N:, self.N:] = neighborhood[:, :, :, :self.N, :self.N].flip((-1,-2))
        expected_block = self.original[:, :, :, i * self.N : (i+2) * self.N, j * self.N : (j+2) * self.N].to(neighborhood)
        #print(neighborhood.shape)
        return neighborhood, expected_block

class lenslet_blocked_referencer(Dataset):
    def __init__(self, decoded, original, MI_size=13):
        super().__init__()
        self.decoded = decoded[0, :1, :, :]
        self.original = original[0, :1, :, :]
        self.N = 32 * MI_size
        self.inner_shape = decoded.shape
        if(decoded.shape == original.shape):
            self.shape = tuple(dim // self.N - 1 for dim in self.inner_shape[-2:])
            assert(all(dim != 0 for dim in self.shape))
        else:
            self.shape = (0,0)
        self.len = fc_reduce(__mul__, self.shape, 1)
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

class reconstructor:
    def __init__(self, shape, N):
        self.N = N
        self.block_shape = tuple(dim // self.N - 1 for dim in shape[-2:])
        self.shape = shape[-5:-2] + tuple(dim * self.N for dim in self.block_shape)
        self.values = np.zeros(self.shape, dtype=np.float32)
        self.i = 0
        self.cap = np.prod(self.block_shape)
    def calculate_coordinate(self, i):
        x = self.i + i
        i, j = (x % self.block_shape[0], x // self.block_shape[0])
        return (i,j)
    def add_blocks(self, blocks):
        new_i = blocks.shape[0] + self.i
        assert(new_i <= self.cap)
        #print(tuple(blocks.shape[1:]))
        #print((*self.shape[:-2], self.N, self.N))
        assert(tuple(blocks.shape[1:]) == (*self.shape[:-2], self.N, self.N))
        #blocks = blocks.astype(np.int16)
        for i in range(blocks.shape[0]):
            x,y = self.calculate_coordinate(i)
            #print(blocks[i, :, :, :, :].shape)
            #print(self.values[:, :, :, x*self.N:(x+1)*self.N, y*self.N:(y+1)*self.N].shape)
            self.values[:, :, :, x*self.N:(x+1)*self.N, y*self.N:(y+1)*self.N] = blocks[i, :, :, :, :]
        self.i = new_i
    def save_image(self, filename):
        #print(self.shape)
        write_LF_PMG(self.values, filename)

    def compare(self, original):
        views_MSE = self.compare_MSE_by_view(original)
        views_PSNR = 20 * np.log(1 / views_MSE ** 0.5)
        PSNR_channel = reduce(views_PSNR, 'c s t -> c', 'mean')
        if len(PSNR_channel) == 1:
            return PSNR_channel[0]
        elif len(PSNR_channel) == 3:
            return (6 * PSNR_channel[0] + 1 * PSNR_channel[1] + 1 * PSNR_channel[2]) / 8

    def compare_MSE_by_view(self, original):
        if original.shape[0] not in (1, 3):
            raise ValueError("Expected LF to have either 1 or 3 color channels, found {original.shape[0]}")
        reference = original[0, :, :, :, self.N:self.N + self.shape[-2], self.N:self.N + self.shape[-1]]
        # print(reference.shape)
        diff = self.values -  reference.numpy()
        squared = np.einsum('...,...->...', diff, diff)
        views_MSE = reduce(squared, 'c s t u v -> c s t', 'mean')
        return views_MSE

class reconstructor_lenslet:
    def __init__(self, shape, N, MI_SIZE=13):
        self.MI_SIZE = MI_SIZE
        self.N = N * MI_SIZE
        self.block_shape = tuple(dim // self.N - 1 for dim in shape[-2:])
        self.shape = (shape[-3], *(dim * self.N for dim in self.block_shape))
        self.values = np.zeros(self.shape, dtype=np.float32)
        self.i = 0
        self.cap = np.prod(self.block_shape)
    def calculate_coordinate(self, i):
        x = self.i + i
        i, j = (x % self.block_shape[0], x // self.block_shape[0])
        return (i,j)
    def add_blocks(self, blocks):
        new_i = blocks.shape[0] + self.i
        assert(new_i <= self.cap)
        assert(tuple(blocks.shape[1:]) == (*self.shape[:-2], self.N, self.N))
        #blocks = blocks.astype(np.int16)
        for i in range(blocks.shape[0]):
            x,y = self.calculate_coordinate(i)
            #print(blocks[i, :, :, :, :].shape)
            #print(self.values[:, :, :, x*self.N:(x+1)*self.N, y*self.N:(y+1)*self.N].shape)
            self.values[:, x*self.N:(x+1)*self.N, y*self.N:(y+1)*self.N] = blocks[i, :, :]
        self.i = new_i
    def save_image(self, filename):
        #print(self.shape)
        write_LF_PNG_lenslet(self.values, filename)
    def compare(self, original):
        views_MSE = self.compare_MSE_by_view(original)
        views_PSNR = 20 * np.log(1 / views_MSE ** 0.5)
        PSNR_channel = reduce(views_PSNR, 'c s t -> c', 'mean')
        if len(PSNR_channel) == 1:
            return PSNR_channel[0]
        elif len(PSNR_channel) == 3:
            return (6 * PSNR_channel[0] + 1 * PSNR_channel[1] + 1 * PSNR_channel[2]) / 8

    def compare_MSE_by_view(self, original):
        if original.shape[0] not in (1, 3):
            raise ValueError("Expected LF to have either 1 or 3 color channels, found {original.shape[0]}")
        reference = original[0, :, self.N:self.N + self.shape[-2], self.N:self.N + self.shape[-1]]
        # print(reference.shape)
        diff = self.values -  reference.numpy()
        squared = np.einsum('...,...->...', diff, diff)
        views_MSE = reduce(squared, 'c (u s) (v t) -> c s t', 'mean', s=self.MI_SIZE, t=self.MI_SIZE)
        return views_MSE
