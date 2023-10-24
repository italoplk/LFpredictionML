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
        (s,t) = (32,32)
        blocker = Rearrange('b c u v (bs s) (bt t) -> b (bs bt c) u v s t', s=32,t=32)
        flatener = Rearrange('b c u v s t -> b c (s t) (u v)', u = u, v = v, s=s, t=t)
        deflatener = Rearrange('b c (s t) (u v)-> b c u v s t', u = u, v = v, s=s, t=t)
        flat_model = UNetLike([ # 18, 1024, 169
            nn.Sequential(
                Conv2d(6, 10, (2,2)), nn.PReLU(), # 10, 1023, 168
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 1022, 167
                Conv2d(10, 10, (1,2)), nn.PReLU(), # 10, 1022, 166
            ),
            nn.Sequential(
                Reduce('b c (x dx) (y dy) -> b c x y', 'max', dx = 2, dy = 2), # 10, 511, 83
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 510, 82
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 509, 81
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 508, 80
            ),
            nn.Sequential(
                Reduce('b c (x dx) (y dy) -> b c x y', 'max', dx = 2, dy = 2), # 10, 254, 40
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 253, 39
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 252, 38
            ),
            nn.Sequential(
                Reduce('b c (x dx) (y dy) -> b c x y', 'max', dx = 3, dy = 2), # 10, 84, 19
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 83, 18
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 82, 17
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 81, 16
            ),
            nn.Sequential(
                Reduce('b c (x dx) (y dy) -> b c x y', 'max', dx = 3, dy = 2), # 10, 27, 8
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 26, 7
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 25, 6
                Conv2d(10, 10, (2,1)), nn.PReLU(), # 10, 24, 6
            ),
            nn.Sequential(
                Reduce('b c (x dx) (y dy) -> b c x y', 'max', dx = 3, dy = 2), # 10, 8, 3
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 7, 2
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 6, 1
                Conv2d(10, 10, (2,1)), nn.PReLU(), # 10, 5, 1
            ),
            nn.Sequential(
                Reduce('b c (x dx) (y dy) -> b c x y', 'max', dx = 5, dy = 1), # 10, 1, 1
                Conv2d(10, 10, (1,1)), nn.PReLU(), # 10, 1, 1
                Conv2d(10, 10, (1,1)), nn.PReLU(), # 10, 1, 1
                Conv2d(10, 10, (1,1)), nn.PReLU(), # 10, 1, 1
            ),
        ], [
            nn.Sequential( # 10, 1, 1
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 5, dy = 1), # 10, 5, 1
            ),
            nn.Sequential( # 10, 5, 2
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 6, 2
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 7, 3
                ConvTranspose2d(10, 10, (2,1)), nn.PReLU(), # 10, 8, 3
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 3, dy = 2), # 10, 24, 6
            ),
            nn.Sequential( # 10, 24, 6
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 25, 7
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 26, 8
                ConvTranspose2d(10, 10, (2,1)), nn.PReLU(), # 10, 27, 8
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 3, dy = 2), # 10, 81, 16
            ),
            nn.Sequential( # 10, 24, 6
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 82, 17
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 83, 18
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 84, 19
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 3, dy = 2), # 10, 252, 38
            ),
            nn.Sequential( # 10, 252, 38
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 253, 39
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 254, 40
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 2, dy = 2), # 10, 508, 80
            ),
            nn.Sequential( # 10, 508, 80
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 509, 81
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 510, 82
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 511, 83
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 2, dy = 2), # 10, 1022, 166
            ),
            nn.Sequential( # 10, 1022, 166
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 1023, 167
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 1024, 168
                ConvTranspose2d(10, 1, (1,2)), nn.PReLU(), # 10, 1024, 169
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
        X = torch.concatenate((X, *flipped), dim = 1)
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

