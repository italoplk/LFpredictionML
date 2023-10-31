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
        flatener = Rearrange('b c u v s t -> b c (u s) (v t)', u = u, v = v)
        deflatener = Rearrange('b c (u s) (v t) -> b c u v s t', u = u, v = v)
        flat_model = UNetLike([ # 18, 832²
            nn.Sequential(
                Conv2d(1, 10, (2,2)), nn.PReLU(), # 10, 831²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 830²
            ),
            nn.Sequential( # 10, 830²
                Reduce('b c (x dx) (y dy) -> b c x y', 'max', dx = 2, dy = 2), # 10, 415²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 414²
            ),
            nn.Sequential( # 10, 414²
                Reduce('b c (x dx) (y dy) -> b c x y', 'max', dx = 2, dy = 2), # 10, 207²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 206²
            ),
            nn.Sequential( # 10, 206²
                Reduce('b c (x dx) (y dy) -> b c x y', 'max', dx=2,dy=2), # 10, 103²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 102²
            ),
            nn.Sequential( # 10, 102²
                Reduce('b c (x dx) (y dy) -> b c x y', 'max', dx=2,dy=2), # 10, 51²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 50²
            ),
            nn.Sequential( # 10, 50²
                Reduce('b c (x dx) (y dy) -> b c x y', 'max', dx=2,dy=2), # 10, 25²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 24²
            ),
            nn.Sequential( # 10, 24²
                Reduce('b c (x dx) (y dy) -> b c x y', 'max', dx=2,dy=2), # 10, 12²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 11²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 10²
            ),
            nn.Sequential( # 10, 10²
                Reduce('b c (x dx) (y dy) -> b c x y', 'max', dx=2,dy=2), # 10, 5²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 4²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 3²
            ),
            nn.Sequential( # 10, 3²
                Reduce('b c (x dx) (y dy) -> b c x y', 'max', dx=3,dy=3), # 10, 1²
            ),
        ], [
            nn.Sequential(
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 2²
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 3²
            ),
            nn.Sequential( # 10, 3²
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 4²
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 5²
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 2, dy = 2), # 10, 10²
            ),
            nn.Sequential( # 10, 10²
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 11²
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 12²
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 2, dy = 2), # 10, 24²
            ),
            nn.Sequential( # 10, 24²
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 25²
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 2, dy = 2), # 10, 50²
            ),
            nn.Sequential( # 10, 50²
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 51²
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 2, dy = 2), # 10, 102²
            ),
            nn.Sequential( # 10, 102²
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 103²
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 2, dy = 2), # 10, 206²
            ),
            nn.Sequential( # 10, 206²
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 207²
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 2, dy = 2), # 10, 414²
            ),
            nn.Sequential( # 10, 414²
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 415²
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 2, dy = 2), # 10, 830²
            ),
            nn.Sequential( # 10, 1, 1
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 831²
                ConvTranspose2d(10, 1, (2,2)), # 10, 832²
            ),
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

