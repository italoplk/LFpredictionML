from torch.nn import Conv2d, ConvTranspose2d
from einops.layers.torch import Rearrange, Reduce
import torch
import torch.nn as nn
import os.path
from unetlike import UNetLike, preserving_dimensions, Repeat
class UNetSpace(nn.Module):
    def __init__(self, name):
        super().__init__()
        u, v = (13,13)
        self.flatener = Rearrange('b c u v s t -> b c (u s) (v t)', u = u, v = v)
        self.deflatener = Rearrange('b c (u s) (v t) -> b c u v s t', u = u, v = v)
        self.flat_model = UNetLike([ # 3, 832, 832
            nn.Sequential(
                Conv2d(3, 10, (5,5)), nn.Dropout(), nn.PReLU(), # 3, 828, 828
            ),
            nn.Sequential( # 10, 828, 828
                Reduce('b c (x dx) (y dy) -> b c x y', 'max', dx = 4, dy = 4), # 10, 207, 207
                Conv2d(10, 10, (4,4)), nn.Dropout(), nn.PReLU(), # 3, 204, 204
                #preserving_dimensions(Conv2d, 10, 10), nn.Dropout(), nn.PReLU()
            ),
            nn.Sequential( # 10, 120, 120
                Reduce('b c (x dx) (y dy) -> b c x y', 'max', dx = 4, dy = 4), # 10, 51, 51
                Conv2d(10, 10, (4,4)), nn.Dropout(), nn.PReLU(), # 10, 48, 48
                #preserving_dimensions(Conv2d, 10, 10), nn.Dropout(), nn.PReLU()
            ),
            nn.Sequential( # 10, 40, 40
                Reduce('b c (x dx) (y dy) -> b c x y', 'max', dx = 4, dy = 4), # 10, 12, 12
                Conv2d(10, 10, (3,3)), nn.Dropout(), nn.PReLU(), # 10, 10, 10
            ),
            nn.Sequential( # 10, 10,  10
                Reduce('b c (x dx) (y dy) -> b c x y', 'max', dx = 10, dy = 10), # 10, 10, 10
            )
        ], [
            nn.Sequential( # 10, 1, 1
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 10, dy = 10), # 10, 10, 10
            ),
            nn.Sequential( # 10, 10, 10
                ConvTranspose2d(20, 10, (3,3)), nn.Dropout(), nn.PReLU(), # 10, 12, 12
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 4, dy = 4) # 10, 48, 48
            ),
            nn.Sequential( # 10, 40, 40
                ConvTranspose2d(20, 10, (4,4)), nn.Dropout(), nn.PReLU(), # 10, 51, 51
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 4, dy = 4) # 10, 204, 204
            ),
            nn.Sequential( # 10, 204, 204
                ConvTranspose2d(20, 10, (4,4)), nn.Dropout(), nn.PReLU(), # 10, 207, 207
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 4, dy = 4) # 10, 828, 828
                #preserving_dimensions(Conv2d, 10, 10), nn.Dropout(), nn.PReLU()
            ),
            nn.Sequential(
              ConvTranspose2d(20, 3, (5,5))
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
zeros = torch.zeros(1, 3, 13, 13, 64, 64)
zeros_t = torch.zeros(1, 3, 13, 13, 64, 64)
lossf = nn.MSELoss()
with torch.no_grad():
    print(lossf(zeros_t, model(zeros)))

