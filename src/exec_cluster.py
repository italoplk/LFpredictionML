
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
from dataset_reader import test_dataset

def random_split_n(data, kfolds):
    random.shuffle(data)
    folds = [ data[foldk::kfolds] for foldk in range(kfolds) ]
    return [(list(itertools.chain.from_iterable(folds[:k] + folds[k+1:])), f) for k, f in enumerate(folds)]
#folds = random_split_n(training_dataset.lfs, 2)


        

import json 
with open("chosen_list.txt", "r") as foldfile:
    folds = json.loads(foldfile.read())
#print(folds)

from torch import save
import torch.nn as nn
from space_only2x2_model import UNetSpace

import wandb




#from space_model_8_small_kernels_stackflip_sum_y import UNetSpace



lossf = nn.MSELoss()

import sys
epochs = 20
batches = (10,)
lr = 1e-5
print('batch: ', batches)


#print(os.listdir('/scratch/Decoded_LFs/png/decoded_32_noPartition'))
#print(os.listdir('/scratch/Original_LFs/png'))




for batch in batches:

    config_saida = f"quick_imgSigmoid_2k_16l.txt"
    print(batch)

    wandb.init(
        # set the wandb project where this run will be logged
        project="predictorUnet",
        # track hyperparameters and run metadata
        name=config_saida,
        config={
            "learning_rate": 0.0005,
            "architecture": f"{config_saida}",
            "dataset": "EPFL",
            "epochs": epochs,
            "BatchSize": batch,
        }
    )


    folder = "/home/shared/MSEs/" + config_saida.split('.txt')[0] + "/"
    os.makedirs(folder, exist_ok=True)



    print(config_saida + "\n", end='', file=open(folder + config_saida, 'w'))


    ##for fold, (training, validation) in enumerate(folds):
    for fold, (training, validation) in enumerate(folds):
        #if fold == 0: continue
        model_name = f"{config_saida}_{fold}"

        folder_train = folder + f"train_fold{fold}/"
        folder_validation = folder + f"validation_fold{fold}/"
        folder_test = folder + f"test_output/"
        os.makedirs(folder_train, exist_ok=True)
        os.makedirs(folder_validation, exist_ok=True)
        os.makedirs(folder_test, exist_ok=True)

        model = UNetSpace(model_name)
        model.cuda()
        print(sum(p.numel() for p in model.parameters() if p.requires_grad))


        optimizer = optim.Adam(model.parameters(), lr)



        for era in range(1,epochs+1):
         #   with open(f"{folder_train}/trainMSE_Views.txt", "a") as outputMSEs:
                # reset file if re-simulating
           #     outputMSEs.write(f"{era}\n")

            f = loop_dataset(functools.partial(train,  model, folder_train, era, fold, config_saida, lossf, optimizer, batch_size=batch, u=1), training)
            save(model.state_dict(), f"model{config_saida}_{era}")


            if (era % 2 == 0):
                print(f"{era}\t{f}", end='', file=open(folder + config_saida, 'a'))
                val = loop_dataset(functools.partial(reconstruct, model, folder_validation, era), validation, { "save_image" : 2} )
                print(f'\t{val}', file=open(folder + config_saida, 'a'))
                print(f'\t{val}')
                wandb.log({"MSE": val})
            else:
                print(f"{era}\t{f}", file=open(folder + config_saida, 'a'))
                print(f"{era}\t{f}")
                wandb.log({"MSE": f})

            # if (era % 10 == 0):
            #     test = loop_dataset(functools.partial(test, model, lossf, optimizer),test)



    # loop_dataset(functools.partial(reconstruct, model, folder_test, 1), test_dataset.lfs)
    loop_dataset(functools.partial(reconstruct, model, folder_test, 1), test_dataset.lfs, {'save_image' : 4}, test_dataset)
    model.save()
    wandb.finish()


