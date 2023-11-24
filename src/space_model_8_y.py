from torch.nn import Conv2d, ConvTranspose2d
from einops.layers.torch import Rearrange, Reduce
import torch
import torch.nn as nn
import os.path
from unetlike import UNetLike, preserving_dimensions, Repeat
class UNetSpace(nn.Module):
    def __init__(self, name):
        super().__init__()
        s, t = (13,13)
        flatener = Rearrange('b c s t u v -> b c (s u) (t v)', s = s, t = t)
        deflatener = Rearrange('b c (s u) (t v) -> b c s t u v', s = s, t = t)
        flat_model = UNetLike([ # 3, 832, 832
            nn.Sequential(
                Conv2d(1, 10, (5,5)), nn.PReLU(), # 3, 828, 828
            ),
            nn.Sequential( # 10, 828, 828
                Reduce('b c (x dx) (y dy) -> b c x y', 'max', dx = 4, dy = 4), # 10, 207, 207
                Conv2d(10, 10, (4,4)), nn.PReLU(), # 3, 204, 204
                #preserving_dimensions(Conv2d, 10, 10), nn.PReLU()
            ),
            nn.Sequential( # 10, 120, 120
                Reduce('b c (x dx) (y dy) -> b c x y', 'max', dx = 4, dy = 4), # 10, 51, 51
                Conv2d(10, 10, (4,4)), nn.PReLU(), # 10, 48, 48
                #preserving_dimensions(Conv2d, 10, 10), nn.PReLU()
            ),
            nn.Sequential( # 10, 40, 40
                Reduce('b c (x dx) (y dy) -> b c x y', 'max', dx = 4, dy = 4), # 10, 12, 12
                Conv2d(10, 10, (3,3)), nn.PReLU(), # 10, 10, 10
            ),
            nn.Sequential( # 10, 10,  10
                Reduce('b c (x dx) (y dy) -> b c x y', 'max', dx = 10, dy = 10), # 10, 10, 10
            )
        ], [
            nn.Sequential( # 10, 1, 1
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 10, dy = 10), # 10, 10, 10
            ),
            nn.Sequential( # 10, 10, 10
                ConvTranspose2d(10, 10, (3,3)), nn.PReLU(), # 10, 12, 12
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 4, dy = 4) # 10, 48, 48
            ),
            nn.Sequential( # 10, 40, 40
                ConvTranspose2d(10, 10, (4,4)), nn.PReLU(), # 10, 51, 51
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 4, dy = 4) # 10, 204, 204
            ),
            nn.Sequential( # 10, 204, 204
                ConvTranspose2d(10, 10, (4,4)), nn.PReLU(), # 10, 207, 207
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 4, dy = 4) # 10, 828, 828
                #preserving_dimensions(Conv2d, 10, 10), nn.PReLU()
            ),
            nn.Sequential(
              ConvTranspose2d(10, 1, (5,5))
            )
        ], compose = lambda x,y: x+y)
        self.f = nn.Sequential(flatener, flat_model, deflatener)
        self.name = name + '.data'
        try:
            if os.path.exists(self.name):
                self.load_state_dict(torch.load(self.name))
        except RuntimeError:
            pass
    
    def save(self):
        torch.save(self.state_dict(), self.name)
    def forward(self, X):
        return self.f(X)[:,:,:,:,32:,32:]

model = UNetSpace("unet_space")
model.eval()
zeros = torch.zeros(1, 1, 13, 13, 64, 64)
zeros_t = torch.zeros(1, 1, 13, 13, 32, 32)
lossf = nn.MSELoss()
with torch.no_grad():
    x = model(zeros)
    print(x.shape)
    print(lossf(zeros_t, x))

