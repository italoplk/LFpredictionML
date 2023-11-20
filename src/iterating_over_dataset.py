import functools
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.utils as utils
import random
import json
from dataset_reader import reconstructor, training_dataset, fold_dataset, blocked_referencer
from torch.utils.data import DataLoader, RandomSampler
import numpy as np
import wandb

lfloader = functools.partial(DataLoader, batch_size=1, num_workers=1, persistent_workers=True)

make_dataloader = functools.partial(DataLoader, batch_size=5, num_workers=2, prefetch_factor=1, persistent_workers=True)


def loop_dataset(action, lfs, mark: Dict[str, int] = dict(), dataset=training_dataset):
    """
        action: função a ser chamada(treinamento ou validação ou teste)... com o LF original, o decodificado, o nome LF, o BPP...
            ... e possivelmente "marcas"(flags True/False determinadas pela entrada mark)
        lfs: uma lista de (classe, nome) de LFs
        mark: parâmetro opcional(vazio por padrão), um dicionário com chaves string e valor um número que diz LFs terão que actions vão receber o parâmetro <chave> com True

        dataset: training_dataset ou testing_dataset
    """
    acc = 0
    i = 0
    # print(len(lfs))
    lfs = fold_dataset(dataset.read_pair, lfs)
    for i, (lf, lfdata) in enumerate(lfs, start=1):
        loader = iter(lfloader(lfdata))
        r = loop_in_lf(action, lf, loader, **{key: i < value for key, value in mark.items()})
        acc += r[0]
        i += r[1]
    return acc / i


def loop_in_lf(action, lf, dataloader, **marks):
    """
        action: função(treinamento) a ser chamada...pelo LF origina, o decodificado, o nome LF, o BPP...
            ... e possivelmente "marcas"(flags True/False determinadas pela marks) como argumentos por nome
        lf: o nome do LF a ser
        dataloader: sequência de LFs definida pelo dataset.read_pair
        **marks: flags a serem passadas em action
    """
    original = next(dataloader)# O primeiro LF da sequência é sempre o original
    acc = 0
    i = 0
    for bpp, decoded in dataloader:
        r = action(original, decoded, lf, bpp, **marks)
        acc += r[0]
        i += r[1]
    # print(f"{lf}\t{acc}")
    return acc, i


def block_MSE_by_view(yt, yc):
    diff = yt - yc
    return torch.einsum('bcuvst,bcuvst->uv', diff, diff) / (diff.shape[-1] * diff.shape[-2])






# auterar o datareader pra sair exemplos
def train(model, folder, era, lossf, optimizer, original, decoded, lf, bpp, batch_size=1, u=1):
    lf = lf
    loader = blocked_referencer(decoded, original)
    acc = 0
    model.train()
    i = 0
    k = 0
    acc_MSE_by_view = 0
    optimizer.zero_grad()
    for inpt, yt in make_dataloader(loader, batch_size=batch_size):
        i += inpt.shape[0]
        # print(inpt.shape)
        # print(yt.shape)
        if torch.cuda.is_available():
            inpt = inpt.cuda()
        y = model(inpt)
        yt = yt[:, :, :, :, 32:, 32:]
        if torch.cuda.is_available():
            yt = yt.cuda()
        err = lossf(yt, y)

        acc_MSE_by_view += block_MSE_by_view(yt, y)


        err.backward()
        k += 1
        if k == u:
            # utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
            k = 0
        err.cpu()
        acc += err.item()
        #MSE DO BATCH
        wandb.log({"acc": acc})
    # Se "sobrou" batches
    if k != 0:
        optimizer.step()
        optimizer.zero_grad()

    MSE_lf = acc / i
    MSE_by_view = acc_MSE_by_view / i



    outfilename = f"{folder}/MSE_Views_train_{'_'.join(lf)}_{''.join(bpp)}"

    data = {'era': era, 'mse_lf': MSE_lf, "mse_by_view": MSE_by_view.cpu().detach().numpy().tolist()}
    with open(f"{outfilename}.json", "w") as outputMSEs:
        # convert tensor to string
        json.dump(data, outputMSEs)

    # model.save()
    return (acc, i)


lossf = nn.MSELoss()


def test(model, original, decoded, *lf):
    lf = lf
    loader = blocked_referencer(decoded, original)
    acc = 0
    model.eval()
    i = 0
    with torch.no_grad() as nograd:
        for inpt, yt in make_dataloader(loader, batch_size=20, num_workers=2, prefetch_factor=5,
                                        persistent_workers=True):
            i += inpt.shape[0]
            inpt = inpt.cuda()
            y = model(inpt)
            yt = yt.cuda()
            err = lossf(yt[:, :, :, :, 32:, 32:], y[:, :, :, :, 32:, 32:]).item()
            acc += err
    return (acc, i)


def reconstruct(model, folder, era, original, decoded, lf, bpp, save_image):
    loader = blocked_referencer(decoded, original)
    i = 0
    acc = 0
    # model.cuda()
    model.eval()
    reconstruc = reconstructor(original.shape, 32)
    for inpt, yt in DataLoader(loader, batch_size=10, num_workers=2, prefetch_factor=5, persistent_workers=True):
        y = model(inpt.cuda() if torch.cuda.is_available() else inpt)
        # y = model(inpt)
        blocks = y
        # err = lossf(yt[:,:,:,:,32:,32:].cuda(), blocks).item()
        yt = yt[:, :, :, :, 32:, 32:]
        yt = yt.cuda() if torch.cuda.is_available() else yt
        err = lossf(yt, blocks).item()
        reconstruc.add_blocks(blocks.cpu().detach().numpy())
        acc += err
        i += y.shape[0]
        # reconstruc.add_blocks(yt[:,:,:,:,32:,32:].cpu().detach().numpy())
        # print(yt.shape[0])
    # print(acc)
    # fprint(reconstruc.i, reconstruc.cap)
    if save_image:
        print(f"{folder}{''.join(lf)}_{''.join(bpp)}")
        reconstruc.save_image(f"{folder}{''.join(lf)}_{''.join(bpp)}")
    # raise StopIteration(acc / i)
    mse_view = reconstruc.compare_MSE_by_view(original)
    mse_lf = reconstruc.compare(original)

    outfilename = f"{folder}/MSE_Views_train_{'_'.join(lf)}_{''.join(bpp)}"

    data = {'era': era, 'mse_lf': float(mse_lf), "mse_by_view": mse_view.tolist()}
    with open(f"{outfilename}.json", "w") as outputMSEs:
        # convert tensor to string
        json.dump(data, outputMSEs)

    return acc, i