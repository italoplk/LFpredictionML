import os.path
from argparse import Namespace

import torch
import torch.nn as nn
from torch.nn import Conv2d, ConvTranspose2d

from Models.unetlike import UNetLike
from einops.layers.torch import Rearrange


class UNetSpace(nn.Module):
    def __init__(self, name, params):
        super().__init__()
        self.s, self.t, self.u, self.v = (params.num_views_ver, params.num_views_hor, params.predictor_size, params.predictor_size)

        self.angle_to_space = Rearrange('... (s u) (t v) -> ... (u s) (v t)', s=self.s,t=self.t)
        self.space_to_angle = Rearrange('... (u s) (v t) -> ... (s u) (t v)', s=self.s,t=self.t)

        flat_model = nn.Sequential(  # 18, 512²
            nn.Sequential(
                Conv2d(1, 10, 3, stride=1), nn.PReLU(),  # 10, 512²
                Conv2d(10, 10, 3, stride=2), nn.PReLU(),  # 10, 254²
            ),
            nn.Sequential(
                Conv2d(10, 10, 3, stride=1), nn.PReLU(),  # 10, 512²
                Conv2d(10, 10, 3, stride=2, padding=1), nn.PReLU(),  # 10, 512²
            ),
            nn.Sequential(
                Conv2d(10, 10, 3, stride=1), nn.PReLU(),  # 10, 512²
                Conv2d(10, 10, 3, stride=2, padding=1), nn.PReLU(),  # 10, 512²
            ),
            nn.Sequential(
                Conv2d(10, 10, 3, stride=1), nn.PReLU(),  # 10, 512²
                Conv2d(10, 10, 3, stride=2, padding=1), nn.PReLU(),  # 10, 512²
            ),
            nn.Sequential(
                Conv2d(10, 10, 3, stride=1), nn.PReLU(),  # 10, 512²
                Conv2d(10, 10, 3, stride=2, padding=1), nn.PReLU(),  # 10, 512²
            ),
            nn.Sequential(  # 10, 510²
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ConvTranspose2d(10, 10, 3, stride=1), nn.PReLU(),  # 1, 512²
            ),
            nn.Sequential(  # 10, 510²
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ConvTranspose2d(10, 10, 3, stride=1), nn.PReLU(),  # 1, 512²
            ),
            nn.Sequential(  # 10, 510²
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ConvTranspose2d(10, 10, 3, stride=1), nn.PReLU(),  # 1, 512²
            ),
            nn.Sequential(  # 10, 510²
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ConvTranspose2d(10, 10, 3, stride=1), nn.PReLU(),  # 1, 512²
            ),
            nn.Sequential(  # 10, 510²
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                ConvTranspose2d(10, 1, 3, stride=1), nn.Sigmoid(),  # 1, 512²
            ),
        )
        self.f = flat_model
        self.name = name + '.data'
        try:
            if os.path.exists(self.name):
                self.load_state_dict(torch.load(self.name))
        except RuntimeError:
            pass
    
    def save(self):
        torch.save(self.state_dict(), self.name)
    def forward(self, X):
        #assert(tuple(X.shape[1:]) == (1,8*64,8*64))
        X = self.angle_to_space(X)
        X = self.f(X)[:,:,-self.s*self.u:,-self.t*self.v:]
        X = self.space_to_angle(X)
        return X



params = Namespace()
dims = (8,8,64,64)
dims_out = (8,8,32,32)
(params.num_views_ver, params.num_views_hor, params.predictor_size, params.predictor_size) = dims_out
# print(params)
model = UNetSpace("unet_space", params)
model.eval()
zeros = torch.zeros(1, 1, dims[0]*dims[2], dims[1]*dims[3])
zeros_t = torch.zeros(1, 1, dims_out[0]*dims_out[2], dims_out[1]*dims_out[3])
lossf = nn.MSELoss()
with torch.no_grad():
    x = model(zeros)
    # print(x.shape)
    print(lossf(zeros_t, x))

