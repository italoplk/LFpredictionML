import torch
import torch.nn as nn
from einops.layers.torch import Reduce


def preserving_dimensions(module_type, channels_in, channels_out):
    return module_type(channels_in, channels_out, 3, 1, 1)

class Repeat(Reduce):
    def __init__(self, pattern, **axes_lengths):
        super().__init__(pattern, 'repeat', **axes_lengths)

class UNetLike(nn.Module):
    def __init__(self, encoder, decoder, compose=lambda x,y: torch.concat((x,y), axis=1)):
        super().__init__()
        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(decoder)
        self.compose = compose
        
    def load(self, path):
        self.load_state_dict(torch.load(path))
    def save(self, path):
        torch.save(self.state_dict(), path)
    def forward(self, X):
        paths = []
        for enc in self.encoder:
            # print(X.shape)
            paths.append(X)
            X = enc(X)
        # print(X.shape)
        X = self.decoder[0](X)
        for s, dec in zip(paths[::-1], self.decoder[1:]):
            #print(X.shape, s.shape)

            X = self.compose(X, s)
            X = dec(X)
        return X
