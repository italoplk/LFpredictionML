
# In[6]:


#concat((torch.zeros(1,2,3), torch.zeros(1,2,3)), axis=1).shape

import os
from torch.utils.data import DataLoader, RandomSampler
#from torchsummary import summary

# In[ ]:


import random
#from torch.utils.data import DataLoader, RandomSampler
import torch.optim as optim
import functools
random.seed(42)


import itertools


from iterating_over_dataset import loop_dataset, reconstruct, train, test


def random_split_n(data, kfolds):
    random.shuffle(data)
    folds = [ data[foldk::kfolds] for foldk in range(kfolds) ]
    return [(list(itertools.chain.from_iterable(folds[:k] + folds[k+1:])), f) for k, f in enumerate(folds)]
#folds = random_split_n(training_dataset.lfs, 2)


        

import json 
with open("chosen_list.txt", "r") as foldfile:
    folds = json.loads(foldfile.read())
#print(folds)


import torch.nn as nn
from space_model_noMax_8l_32in64out_correctingKernels_L1K2x2 import UNetSpace

epochs = 1000

#from space_model_8_small_kernels_stackflip_sum_y import UNetSpace



lossf = nn.MSELoss()

import sys
batches = (10,)
lr = 1e-5
print('batch: ', batches)


#print(os.listdir('/scratch/Decoded_LFs/png/decoded_32_noPartition'))
#print(os.listdir('/scratch/Original_LFs/png'))




for batch in batches:

    configSaida = f"space_8_correctedFolds_L1-2k_4k_noMPnoDR_HBPP_32in64ou_B{batch}_LR{lr}.txt"
    print(batch)
    print(f"modelname: {configSaida}", file=open('/scratch/' + configSaida, 'w'))
    ##for i, (training, validation) in enumerate(folds):
    for i, (training, validation) in enumerate(folds):
        #if i == 0: continue
        model_name = f"{configSaida}_{i}"
        model = UNetSpace(model_name)
        model.cuda()
        print(sum(p.numel() for p in model.parameters() if p.requires_grad))


        optimizer = optim.Adam(model.parameters(), lr)
        for era in range(1,epochs+1):
            f = loop_dataset(functools.partial(train, model, lossf, optimizer, batch_size=10, u=2), training)
            if (era % 20 == 0):
                print(f"{era}\t{f}", end='', file=open('/scratch/'+ configSaida, 'a'))
                folder = f"{model_name}_examples/{era}/"
                os.makedirs(folder, exist_ok=True)
                val = loop_dataset(functools.partial(reconstruct, model, folder), validation[:2])
                print(f'\t{val}', file=open('/scratch/'+ configSaida, 'a'))
            else:
                print(f"{era}\t{f}", file=open('/scratch/'+ configSaida, 'a'))

            # if (era % 10 == 0):
            #     test = loop_dataset(functools.partial(test, model, lossf, optimizer),test)

    model.save()


