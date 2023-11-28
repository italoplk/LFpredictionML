from DataSet import DataSet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:

    def __init__(self, dataset: DataSet, config_name, params):

        self.model_name = params.model
        # TODO make loss GREAT AGAIN, nope, make it a param.
        self.loss = nn.MSELoss()

        # TODO after everything else is done, adapt for other models
        if self.model_name == 'Unet2k':
            try:
                from Models.space_only2x2_model import UNetSpace
                # talvez faça mais sentido sò passar as variaveis necessarias do dataset
                model = UNetSpace(config_name, params)
            except RuntimeError as e:
                print("Failed to import model: ", e)

            # TODO make AMERICA GREAT AGAIN, nope.... Num works be a parameter too
            self.train_set = DataLoader(dataset.list_train, shuffle=True, batch_size=params.batch_size, num_workers=8,
                                   pin_memory=True)
            self.val_set = DataLoader(dataset.list_test, shuffle=False, batch_size=params.batch_size, num_workers=8,
                                 pin_memory=True)
            self.test_set = DataLoader(dataset.list_test, shuffle=False, batch_size=1, num_workers=8, pin_memory=True)

            if torch.cuda.is_available():
                model = model.cuda()
                device = torch.device("cuda")
            else:
                print("Running on CPU!")
                device = torch.device("cpu")

            self.loss = self.loss.to(device)

            # TODO check betas
            optimizer = torch.optim.Adam(
                model.parameters(), lr=params.lr, betas=(0.9, 0.999))


            for epoch in params.epochs:
                loss = self.train(epoch)

    def train(self, epoch):
        acc = 0

        for i, data in enumerate(self.train_set):


            self.lf.load_lf()

            self.lf.get_block()


            self.model.train()
