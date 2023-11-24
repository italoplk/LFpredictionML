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
        blocker = Rearrange('b c s t (bu u) (bv v) -> b (bu bv c) s t u v', u=32,v=32)
        flatener = Rearrange('b c s t u v -> b c (s u) (t v)', s = s, t = t)
        deflatener = Rearrange('b c (s u) (t v) -> b c s t u v', s = s, t = t)
        flat_model = UNetLike([ # 18, 416²
            nn.Sequential(
                Conv2d(6, 10, (2,2)), nn.PReLU(), # 10, 415²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 414²
            ),
            nn.Sequential( # 10, 414²
                Reduce('b c (x dx) (y dy) -> b c x y', 'max', dx = 2, dy = 2), # 10, 207²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 206²
            ),
            nn.Sequential( # 10, 206²
                Reduce('b c (x dx) (y dy) -> b c x y', 'max', dx = 2, dy = 2), # 10, 103²
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
            nn.Sequential( # 10, 10,  10
                Reduce('b c (x dx) (y dy) -> b c x y', 'max', dx=2,dy=2), # 10, 12²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 11²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 10²
            ),
            nn.Sequential( # 10, 10,  10
                Reduce('b c (x dx) (y dy) -> b c x y', 'max', dx=2,dy=2), # 10, 5²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 4²
            ),
            nn.Sequential( # 10, 10,  10
                Reduce('b c (x dx) (y dy) -> b c x y', 'max', dx=2,dy=2), # 10, 2²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 1²
            ),
        ], [
            nn.Sequential( # 10, 1²
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 2²
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 2, dy = 2), # 10, 4²
            ),
            nn.Sequential( # 10, 1, 1
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 5²
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 2, dy = 2), # 10, 4²
            ),
            nn.Sequential( # 10, 1, 1
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 5²
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 5²
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 2, dy = 2), # 10, 4²
            ),
            nn.Sequential( # 10, 1, 1
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 5²
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 2, dy = 2), # 10, 4²
            ),
            nn.Sequential( # 10, 1, 1
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 5²
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 2, dy = 2), # 10, 4²
            ),
            nn.Sequential( # 10, 1, 1
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 5²
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 2, dy = 2), # 10, 4²
            ),
            nn.Sequential( # 10, 1, 1
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 5²
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 2, dy = 2), # 10, 4²
            ),
            nn.Sequential( # 10, 1, 1
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 5²
                ConvTranspose2d(10, 1, (2,2)), nn.Sigmoid(), # 10, 5²
            ),
        ], compose = lambda x,y: x+y)
        self.blocker = blocker
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
        flip = torch.flip
        X = self.blocker(X)[:, :3, :, :, :, :]
        d, u, l = X[:, :1, :, :, :, :], X[:, 1:2, :, :, :, :], X[:, 2:3, :, :, :, :]
        flipped = map(flip, (d, u, l), ((-2,-1), (-2,), (-1,)))
        X = torch.cat((X, *flipped), dim = 1)
        #print(X.shape)
        return self.f(X)

model = UNetSpace("unet_space")
model.eval()
zeros = torch.zeros(1, 1, 13, 13, 64, 64)
zeros_t = torch.zeros(1, 1, 13, 13, 32, 32)
lossf = nn.MSELoss()
with torch.no_grad():
    x = model(zeros)
    print(x.shape)
    print(lossf(zeros_t, x))

