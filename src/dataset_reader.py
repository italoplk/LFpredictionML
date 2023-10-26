from glob import glob
#print(glob("Original_LFs/mat/*/*")[:5])
#print(glob("Decoded_LFs/mat/*/*/*")[:5])


# In[2]:


#import h5py
from glob import iglob
import os
import numpy as np
import torch
from random import shuffle, sample
from torch.utils.data import DataLoader
from itertools import accumulate
from einops import rearrange
import cv2
import einops

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
import sys
from einops import EinopsError

normalizer_factor = 2/(2 ** 16 - 1)
def read_LF_PNG(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)

def normalize_16bit_image(image):
    return torch.tensor(image.astype(np.float32)) * normalizer_factor - 1

def unnormalize_to_16bit_image(image):
    return (((image +1)/normalizer_factor).astype(np.uint16))

#write the LFs after validation
def write_LF_PMG(image, path):
    image = rearrange(image, 'c s t u v -> (s u) (t v) c', s=13, t=13)
    image = unnormalize_to_16bit_image(image)
    #image = cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)
    #print(image.shape)
    cv2.imwrite(f'{path}.png', image)

def read_LF(path):
    img = read_LF_PNG(path)
    img_ycbcr = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB))
    img_normalized = normalize_16bit_image(img_ycbcr)
    try:
        return rearrange(img_normalized, '(s u) (t v) c -> c s t u v', s=13, t=13)[:1, :, :, :, :]
    except EinopsError:
        print(f"failed to read {path}", file=sys.stderr)
        return np.zeros((3,1,1,1,1))

on_the_list = lambda candidate, l: any(test in candidate for test in l)
#modificar pairwise para retornar uma lista de blocos
class pairwise_lister:
    def __init__(self, originals, decoded, exin_clude, exclude = True):
        self.decoded = decoded
        self.originals = originals
        if exclude:
            included = lambda candidate: not(on_the_list(candidate, exin_clude))
        else:
            included = lambda candidate: on_the_list(candidate, exin_clude)
        
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
    

    def read_original_mat(self, lfclass, lf):
        #path = '/'.join((self.originals, lfclass, lf.replace("___", "_&_") + '.mat.png'))
        path = '/'.join((self.originals, lfclass, lf + '.mat.png'))
        return read_LF(path)
        #with h5py.File(path, 'r') as f:
        #    return np.array(f['dataset'])
    
    def read_decoded_mat(self, lfclass, lf, bpp):
        path = '/'.join((self.decoded, lfclass, lf, bpp))
        return (bpp, read_LF(path))
    
    def read_pair(self, lfclass, lf):
        bpps = list(self.bbps[lfclass, lf])
        shuffle(bpps)
        original = single(self.read_original_mat, (lfclass, lf))
        #print(len(original))
        decoded = partial(self.read_decoded_mat, lfclass, lf)
        data = chain(original, mymap(decoded, bpps))
        return data

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


training_dataset = pairwise_lister("/scratch/Original_LFs/png", "/scratch/HBPP/", ["Bikes", "Danger_de_Mort", "Fountain___Vincent_2", "Stone_Pillars_Outside"], exclude = True)

test_dataset = pairwise_lister("/scratch/Original_LFs/png", "/scratch/HBPP/", ["Bikes", "Danger_de_Mort", "Fountain___Vincent_2", "Stone_Pillars_Outside"], exclude = False)



#display(list((k, v) for k,v in training_dataset.bbps.items() if len(v) != 5)[:5])


# In[3]:


#torch.tensor([0,1,2,3], dtype=torch.float32)


# In[4]:


from functools import reduce, partial
from operator import __mul__
import torch
from torch.utils.data import Dataset
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
        self.len = reduce(__mul__, self.shape, 1)
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
        folder = '/scratch/'.join(filename.split('/')[:-1])
        write_LF_PMG(self.values, filename)

    def compare(self, original):
        views_MSE = self.compare_MSE_by_view(original)
        views_PSNR = 20 * np.log(1 / views_MSE ** 0.5)
        PSNR_channel = einops.reduce(views_PSNR, 'c s t -> c', 'mean')
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
        views_MSE = einops.reduce(squared, 'c s t u v -> c s t', 'mean')
        return views_MSE
