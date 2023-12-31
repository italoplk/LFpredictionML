
# EXEC_TRAIN.py but prints in CMD


#concat((torch.zeros(1,2,3), torch.zeros(1,2,3)), axis=1).shape

import os

import torch.cuda
from torch.utils.data import DataLoader, RandomSampler
#from torchsummary import summary

# In[ ]:


import random
#from torch.utils.data import DataLoader, RandomSampler
import torch.optim as optim
import functools
random.seed(42)


import itertools

from iterating_over_dataset_local import loop_dataset, reconstruct, train, test
from dataset_reader import test_dataset
import torch.nn as nn


def random_split_n(data, kfolds):
    random.shuffle(data)
    folds = [ data[foldk::kfolds] for foldk in range(kfolds) ]
    return [(list(itertools.chain.from_iterable(folds[:k] + folds[k+1:])), f) for k, f in enumerate(folds)]
#folds = random_split_n(training_dataset.lfs, 2)


        

import json 
with open("chosen_list.txt", "r") as foldfile:
    folds = json.loads(foldfile.read())
#print(folds)



epochs = 10

configSaida = 'test.txt'

#from space_model_8_small_kernels_stackflip_sum_y import UNetSpace

#from space_model_noMax_smallerKernel_8l import UNetSpace
from space_model_noMax_8l_32in64out_correctingKernels_L1K2x2 import UNetSpace

if(not torch.cuda.is_available()):
    print(torch.cuda.is_available())
    #exit()

lossf = nn.MSELoss()

import sys
batches = (10,)
print(batches)
#print(os.listdir('/scratch/Decoded_LFs/png/decoded_32_noPartition'))
#print(os.listdir('/scratch/Original_LFs/png'))

for batch in batches:
    print(batch)
    ##for i, (training, validation) in enumerate(folds):
    for i, (training, validation) in enumerate(folds):
        print(validation)
        #if i == 0: continue
        model_name = f"space8_batch_{batch}_{i}"
        model = UNetSpace(model_name)
        model.cuda()
        print(sum(p.numel() for p in model.parameters() if p.requires_grad))

        
        optimizer = optim.Adam(model.parameters(), lr = 1e-4)



        for era in range(1, epochs+1):
            f = loop_dataset(functools.partial(train, model, lossf, optimizer, batch_size=10, u=2), training)
            if (era % 10 == 0):
                print(f"{era}\t{f}", end='', file=open(configSaida, 'a'))
                folder = f"{model_name}_examples/{era}/"
                os.makedirs(folder, exist_ok=True)
                val = loop_dataset(functools.partial(reconstruct, model, folder), validation[:2])
                print(f'\t{val}', file=open(configSaida, 'a'))
            else:
                print(f"{era}\t{f}", file=open(configSaida, 'a'))

            # if era % 1 == 0:


        
# In[ ]:


