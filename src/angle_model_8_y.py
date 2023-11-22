from torch.nn import Conv2d, ConvTranspose2d
from einops.layers.torch import Rearrange, Reduce
import torch
import torch.nn as nn
import os.path
from unetlike import UNetLike, preserving_dimensions, Repeat
class UNetAngle(nn.Module):
    def __init__(self, name):
        super().__init__()
        u, v = (13,13)
        #blocker = Rearrange('b c u v (bs s) (bt t) -> b (bs bt c) u v s t', s=32,t=32)
        #flatener = Rearrange('b c u v s t -> b c (u s) (v t)', u = u, v = v)
        #deflatener = Rearrange('b c (u s) (v t) -> b c u v s t', u = u, v = v)
        flat_model = UNetLike([ # 1, 576, 576
            nn.Sequential(
                Conv2d(1, 10, (5,5)), nn.PReLU(), # 10, 576, 576
            ),
            nn.Sequential( # 10, 572, 572
                Reduce('b c (x dx) (y dy) -> b c x y', 'max', dx = 4, dy = 4), # 10, 143, 143
                Conv2d(10, 10, (4,4)), nn.PReLU(), # 10, 140, 140
                #preserving_dimensions(Conv2d, 10, 10), nn.PReLU()
            ),
            nn.Sequential( # 10, 140, 140
                Reduce('b c (x dx) (y dy) -> b c x y', 'max', dx = 4, dy = 4), # 10, 35, 35
                Conv2d(10, 10, (4,4)), nn.PReLU(), # 10, 32, 32
                #preserving_dimensions(Conv2d, 10, 10), nn.PReLU()
            ),
            nn.Sequential( # 10, 32, 32
                Reduce('b c (x dx) (y dy) -> b c x y', 'max', dx = 4, dy = 4), # 10, 8, 8
                Conv2d(10, 10, (3,3)), nn.PReLU(), # 10, 6,6
            ),
            nn.Sequential( # 10, 6, 6
                Reduce('b c (x dx) (y dy) -> b c x y', 'max', dx = 6, dy = 6), # 10, 1,1
            )
        ], [
            nn.Sequential( # 10, 1, 1
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 6, dy = 6), # 10, 6, 6
            ),
            nn.Sequential( # 10, 6, 6
                ConvTranspose2d(10, 10, (3,3)), nn.PReLU(), # 10, 8, 8
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 4, dy = 4) # 10, 32, 32
            ),
            nn.Sequential( # 10, 32, 32
                ConvTranspose2d(10, 10, (4,4)), nn.PReLU(), # 10, 35, 35
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 4, dy = 4) # 10, 140, 140
            ),
            nn.Sequential( # 10, 140, 140
                ConvTranspose2d(10, 10, (4,4)), nn.PReLU(), # 10, 143, 143
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 4, dy = 4) # 10, 572, 572
                #preserving_dimensions(Conv2d, 10, 10), nn.PReLU()
            ),
            nn.Sequential(
              ConvTranspose2d(10, 1, (5,5)) # 10, 576, 576
            )
        ], compose = lambda x,y: x+y)
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
        return self.f(X)[:,:,9*32:,9*32:]

model = UNetAngle("unet_angle")
model.eval()
zeros = torch.zeros(1, 1, 9 * 64, 9 * 64)
zeros_t = torch.zeros(1, 1, 9 * 32, 9 * 32)
lossf = nn.MSELoss()
with torch.no_grad():
    x = model(zeros)
    print(x.shape)
    print(lossf(zeros_t, x))

