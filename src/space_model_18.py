from torch.nn import Conv2d, ConvTranspose2d
from einops.layers.torch import Rearrange
import torch
import torch.nn as nn
import os.path
from unetlike import UNetLike, preserving_dimensions
class UNetSpace(nn.Module):
    def __init__(self, name):
        super().__init__()
        u, v = (13,13)
        self.flatener = Rearrange('b c u v s t -> b c (u s) (v t)', u = u, v = v)
        self.deflatener = Rearrange('b c (u s) (v t) -> b c u v s t', u = u, v = v)
        self.flat_model = UNetLike([ # 4, 832, 832
            nn.Sequential(
              Conv2d(4, 10, (20,20), stride = (5,5), padding=(4,4)), nn.Dropout(), nn.PReLU(),
              preserving_dimensions(Conv2d, 10, 10), nn.Dropout(), nn.PReLU(),
              preserving_dimensions(Conv2d, 10, 10), nn.Dropout(), nn.PReLU()
            ),
            nn.Sequential( # 10, 163, 163
              Conv2d(10, 10, (15,15), stride = (5,5), padding=(1,1)), nn.Dropout(), nn.PReLU(),
              preserving_dimensions(Conv2d, 10, 10), nn.Dropout(), nn.PReLU(),
              preserving_dimensions(Conv2d, 10, 10), nn.Dropout(), nn.PReLU()
            ),
            nn.Sequential( # 10, 31, 31
              Conv2d(10, 10, (31,31), stride = (1,1)), nn.Dropout(), nn.PReLU()
            ),
        ], [
            nn.Sequential(
              ConvTranspose2d(10, 10, (31,31), stride = (1,1)), nn.Dropout(), nn.PReLU(),
              preserving_dimensions(Conv2d, 10, 10), nn.Dropout(), nn.PReLU(),
              preserving_dimensions(Conv2d, 10, 10), nn.Dropout(), nn.PReLU()
            ),
            nn.Sequential(
              ConvTranspose2d(20, 10, (15,15), stride = (5,5)), nn.Dropout(), nn.PReLU(),
              preserving_dimensions(Conv2d, 10, 10), nn.Dropout(), nn.PReLU(),
              preserving_dimensions(Conv2d, 10, 10), nn.Dropout(), nn.PReLU()
            ),
            nn.Sequential(
              ConvTranspose2d(20, 10, (20,20), stride = (5,5), padding=(4,4)),
              preserving_dimensions(Conv2d, 10, 10), nn.Dropout(), nn.PReLU(),
              preserving_dimensions(Conv2d, 10, 10), nn.Dropout(), nn.PReLU()
            ),
            nn.Sequential(
              preserving_dimensions(Conv2d, 14, 10), nn.Dropout(), nn.PReLU(),
              preserving_dimensions(Conv2d, 10, 3),
            )
        ])
        
        self.name = name + '.data'
        try:
            if os.path.exists(self.name):
                self.load_state_dict(torch.load(self.name))
        except RuntimeError:
            pass
    
    def save(self):
        torch.save(self.state_dict(), self.name)
    def forward(self, X):
        return self.deflatener(self.flat_model(self.flatener(X)))

model = UNetSpace("unet_space")
model.eval()
zeros = torch.zeros(1, 4, 13, 13, 64, 64)
zeros_t = torch.zeros(1, 3, 13, 13, 64, 64)
lossf = nn.MSELoss()
with torch.no_grad():
    print(lossf(zeros_t, model(zeros)))

