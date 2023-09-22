
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

epochs = 10

#from space_model_8_small_kernels_stackflip_sum_y import UNetSpace

from space_model_noMax_8l import UNetSpace

lossf = nn.MSELoss()

import sys
batches = (160,) 
print(batches)
for batch in batches:
    print(batch)
    ##for i, (training, validation) in enumerate(folds):
    for i, (training, validation) in enumerate(folds):
        #if i == 0: continue
        model_name = f"space8_batch_{batch}_{i}"
        model = UNetSpace(model_name)
        model.cuda()
        print(sum(p.numel() for p in model.parameters() if p.requires_grad))


        
        
        optimizer = optim.Adam(model.parameters(), lr = 1e-4)
        for era in range(1,epochs+1):
            f = loop_dataset(functools.partial(train, model, lossf, optimizer, batch_size=10, u=2), training)
            if (era % 20 == 0):
                print(f"{era}\t{f}", end='', file=open('output.txt', 'a'))
                folder = f"{model_name}_examples/{era}/"
                os.makedirs(folder, exist_ok=True)
                val = loop_dataset(functools.partial(reconstruct, model, folder), validation[:2])
                print(f'\t{val}', file=open('output.txt', 'a'))
            else:
                print(f"{era}\t{f}", file=open('output.txt', 'a'))
        
# In[ ]:


