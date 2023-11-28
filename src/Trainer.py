from argparse import Namespace
from DataSet import DataSet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,BlockedReferencerLenslet
from tqdm import tqdm


class Trainer:

    def __init__(self, dataset: DataSet, config_name : str, params : Namespace):
        self.model_name = params.model
        # TODO make loss GREAT AGAIN, nope, make it a param.
        self.loss = nn.MSELoss()
        self.params = params

        # TODO after everything else is done, adapt for other models
        self.model = ModelOracle(params.model).get_model(config_name, params)
        # TODO make AMERICA GREAT AGAIN, nope.... Num works be a parameter too
        # TODO test prefetch_factor and num_workers to optimize
        self.train_set = DataLoader(dataset.list_train, shuffle=True, num_workers=1,
                                pin_memory=True, prefetch_factor=2)
        self.val_set = DataLoader(dataset.list_test, shuffle=False, num_workers=8,
                                pin_memory=True)
        self.test_set = DataLoader(dataset.list_test, shuffle=False, num_workers=8, pin_memory=True)

        if torch.cuda.is_available():
            self.model.cuda()
            device = torch.device("cuda")
        else:
            print("Running on CPU!")
            device = torch.device("cpu")

        self.loss = self.loss.to(device)

        # TODO check betas
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=params.lr, betas=(0.9, 0.999))


        for epoch in range(1, 1+params.epochs):
            loss = self.train(epoch)

    def train(self, current_epoch):
        acc = 0

        for i, data in enumerate(self.train_set):


            self.lf.load_lf()
            self.lf.get_block()
            self.model.train()

class ModelOracle:
    def __init__(self, model_name):
        if model_name == 'Unet2k':
            from Models.space_only2x2_model import UNetSpace
            # talvez faça mais sentido sò passar as variaveis necessarias do dataset
            self.model = UNetSpace
        elif model_name == 'Unet3k':
            from Models.space_model_8_small_kernels import UNetSpace
            self.model = UNetSpace
        else:
            print("Model not Found.")
            exit(404)

    def get_model(self, config_name, params):
        try:
            return self.model(config_name, params)
        except RuntimeError as e:
            print("Failed to import model: ", e)

