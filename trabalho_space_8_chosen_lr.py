
# In[6]:


#torch.concat((torch.zeros(1,2,3), torch.zeros(1,2,3)), axis=1).shape

import os
from torch.utils.data import DataLoader, RandomSampler


# In[ ]:


import random
#from torch.utils.data import DataLoader, RandomSampler
import torch.optim as optim
import functools
random.seed(42)


import itertools

def random_split_n(data, kfolds):
    random.shuffle(data)
    folds = [ data[foldk::kfolds] for foldk in range(kfolds) ]
    return [(list(itertools.chain.from_iterable(folds[:k] + folds[k+1:])), f) for k, f in enumerate(folds)]
#folds = random_split_n(training_dataset.lfs, 2)


        

import json 
with open("chosen_list.txt", "r") as foldfile:
    folds = json.loads(foldfile.read())
#print(folds)

from iterating_over_dataset import loop_dataset, reconstruct, train, test
import torch.nn as nn

epochs = 800

from space_model_8_small_kernels_stackflip_sum_y import UNetSpace

import sys
lrs = (1e-4,) if (len(sys.argv) < 2) else tuple(map(float, sys.argv[1:]))
print(lrs)
#for lr in lrs:

lossf = nn.MSELoss()
for lr in lrs:
    print(lr)
    for i, (training, validation) in enumerate(folds):
        if i == 0: continue
        model_name = f"space8_lr_{lr}_{i}"
        model = UNetSpace(model_name)
        model.cuda()
        optimizer = optim.Adam(model.parameters(), lr = lr)
        for era in range(1, epochs+1):
            f = loop_dataset(functools.partial(train, model, lossf, optimizer, batch_size = 10), training)
            if (era % 20 == 0):
                print(f"{era}\t{f}", end='')
                folder = f"{model_name}_examples/{era}/"
                os.makedirs(folder, exist_ok=True)
                val = loop_dataset(functools.partial(reconstruct, model, folder), validation[:2])
                print(f'\t{val}')            
            else:
                print(f"{era}\t{f}")
        
# In[ ]:


