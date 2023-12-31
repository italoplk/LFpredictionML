import functools
import torch
import torch.nn as nn
import torch.nn.utils as utils
import random
from dataset_reader_local import reconstructor, training_dataset, fold_dataset, blocked_referencer
from torch.utils.data import DataLoader, RandomSampler

lfloader = functools.partial(DataLoader, batch_size = 1, num_workers = 1, persistent_workers = True)

make_dataloader = functools.partial(DataLoader, batch_size = 5, num_workers = 2, prefetch_factor = 1, persistent_workers = True)

def loop_dataset(action, lfs):
    acc = 0
    i = 0
    #print(len(lfs))
    lfs = fold_dataset(training_dataset.read_pair, lfs)
    for lf, lfdata in lfs:
        loader = iter(lfloader(lfdata))
        r = loop_in_lf(action, lf, loader)
        acc += r[0]
        i += r[1]
    return acc/i

def loop_in_lf(action, lf, dataloader):
    original = next(dataloader)
    acc = 0
    i = 0
    for bpp,decoded in dataloader:
        r = action(original, decoded, lf, bpp)
        acc += r[0]
        i += r[1]
    #print(f"{lf}\t{acc}")
    return acc, i

#auterar o datareader pra sair exemplos
def train(model, lossf, optimizer, original, decoded, *lf, batch_size = 1, u=0):
    lf = lf 
    loader = blocked_referencer(decoded, original)
    acc = 0
    model.train()
    i = 0
    k = 0
    for inpt, yt in make_dataloader(loader, batch_size = batch_size):
        i += inpt.shape[0]
        #print(inpt.shape)
        #print(yt.shape)
        inpt=inpt.cuda()
        y = model(inpt)
        yt=yt.cuda()
        err = lossf(yt[:,:,:,:,32:,32:], y)
        if k == u:
            err.backward()
            #utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
            k = 0
        else:
            k +=  1
        err.cpu()
        acc += err.item()
    model.save()
    return (acc,i)


lossf = nn.MSELoss()
def test(model, original, decoded, *lf):
    lf = lf
    loader = blocked_referencer(decoded, original)
    acc = 0
    model.eval()
    i = 0
    with torch.no_grad() as nograd:
        for inpt, yt in make_dataloader(loader, batch_size = 20, num_workers = 2, prefetch_factor = 5, persistent_workers = True):
            i += inpt.shape[0]
            inpt=inpt.cuda()
            y = model(inpt)
            yt=yt.cuda()
            err = lossf(yt[:,:,:,:,32:,32:], y[:,:,:,:,32:,32:]).item()
            acc += err
    return (acc, i)

def reconstruct(model, prefix, original, decoded, lf, bpp):
    loader = blocked_referencer(decoded, original)
    i = 0
    acc = 0
    #model.cuda()
    model.eval()
    reconstruc = reconstructor(original.shape, 32)
    for inpt, yt in DataLoader(loader, batch_size = 10, num_workers = 2, prefetch_factor = 5, persistent_workers = True):
        y = model(inpt.cuda())
        #y = model(inpt)
        blocks = y
        #err = lossf(yt[:,:,:,:,32:,32:].cuda(), blocks).item()
        err = lossf(yt[:,:,:,:,32:,32:].cuda(), blocks).item()
        reconstruc.add_blocks(blocks.cpu().detach().numpy())
        acc += err
        i += y.shape[0]
        #reconstruc.add_blocks(yt[:,:,:,:,32:,32:].cpu().detach().numpy())
        #print(yt.shape[0])
    #print(acc)
    #fprint(reconstruc.i, reconstruc.cap)
    reconstruc.save_image(f"{prefix}{''.join(lf)}_{''.join(bpp)}")
    #raise StopIteration(acc / i)
    return acc, i
