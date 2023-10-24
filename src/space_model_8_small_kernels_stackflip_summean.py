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
        blocker = Rearrange('b c u v (bs s) (bt t) -> b (bs bt c) u v s t', s=32,t=32)
        flatener = Rearrange('b c u v s t -> b c (u s) (v t)', u = u, v = v)
        deflatener = Rearrange('b c (u s) (v t) -> b c u v s t', u = u, v = v)
        flat_model = UNetLike([ # 18, 416, 416
            nn.Sequential(
                preserving_dimensions(Conv2d, 18, 10), nn.Dropout(), nn.PReLU() # 10, 416, 416
            ),
            nn.Sequential( # 10, 828, 828
                Reduce('b c (x dx) (y dy) -> b c x y', 'mean', dx = 4, dy = 4), # 10, 104, 104
                preserving_dimensions(Conv2d, 10, 10), nn.Dropout(), nn.PReLU(), # 10, 104, 104
                #preserving_dimensions(Conv2d, 10, 10), nn.Dropout(), nn.PReLU()
            ),
            nn.Sequential( # 10, 120, 120
                Reduce('b c (x dx) (y dy) -> b c x y', 'mean', dx = 4, dy = 4), # 10, 26, 26
                Conv2d(10, 10, (3,3)), nn.Dropout(), nn.PReLU(), # 10, 24, 24
                #preserving_dimensions(Conv2d, 10, 10), nn.Dropout(), nn.PReLU()
            ),
            nn.Sequential( # 10, 40, 40
                Reduce('b c (x dx) (y dy) -> b c x y', 'mean', dx = 4, dy = 4), # 10, 6, 6
                Conv2d(10, 10, (3,3)), nn.Dropout(), nn.PReLU(), # 10, 4, 4
            ),
            nn.Sequential( # 10, 10,  10
                Reduce('b c (x dx) (y dy) -> b c x y', 'mean', dx = 4, dy = 4), # 10, 1, 1
            )
        ], [
            nn.Sequential( # 10, 1, 1
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 4, dy = 4), # 10, 4, 4
            ),
            nn.Sequential( # 10, 10, 10
                ConvTranspose2d(10, 10, (3,3)), nn.Dropout(), nn.PReLU(), # 10, 6, 6
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 4, dy = 4) # 10, 24, 24
            ),
            nn.Sequential( # 10, 40, 40
                ConvTranspose2d(10, 10, (3,3)), nn.Dropout(), nn.PReLU(), # 10, 26, 26
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 4, dy = 4) # 10, 104, 104
            ),
            nn.Sequential( # 10, 204, 204
                preserving_dimensions(Conv2d, 10, 10), nn.Dropout(), nn.PReLU(), # 10, 104, 104
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 4, dy = 4) # 10, 416, 416
                #preserving_dimensions(Conv2d, 10, 10), nn.Dropout(), nn.PReLU()
            ),
            nn.Sequential(
                preserving_dimensions(Conv2d, 10, 3)
            )
        ], compose = lambda x,y: x+y) # type: ignore
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
        X = self.blocker(X)[:, :9, :, :, :, :]
        d, u, l = X[:, :3, :, :, :, :], X[:, 3:6, :, :, :, :], X[:, 6:9, :, :, :, :]
        flipped = map(flip, (d, u, l), ((-2,-1), (-2,), (-1,)))
        X = torch.concat((X, *flipped), dim = 1)
        #print(X.shape)
        return self.f(X)

model = UNetSpace("unet_space")
model.eval()
zeros = torch.zeros(1, 3, 13, 13, 64, 64)
zeros_t = torch.zeros(1, 3, 13, 13, 32, 32)
lossf = nn.MSELoss()
with torch.no_grad():
    print(lossf(zeros_t, model(zeros)))

