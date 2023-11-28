from torch.nn import Conv2d, ConvTranspose2d
from einops.layers.torch import Rearrange, Reduce
import torch
import torch.nn as nn
import os.path
from Models.unetlike import UNetLike, preserving_dimensions, Repeat
class UNetSpace(nn.Module):
    def __init__(self, name, params):
        super().__init__()
        s, t, u, v = (params['views_h'], params['views_w'], params['divider_block_h'], params['divider_block_w'])
        blocker = Rearrange('b c s t (bu u) (bv v) -> b (bu bv c) s t u v', u=u,v=v)
        flatener = Rearrange('b c s t u v -> b c (s u) (t v)', s = s, t = t)
        deflatener = Rearrange('b c (s u) (t v) -> b c s t u v', s = s, t = t)
        flat_model = UNetLike([ # 18, 288²
            nn.Sequential(
                Conv2d(6, 10, (2,2)), nn.PReLU(), # 10, 287²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 286²
            ),
            nn.Sequential( # 10, 414²
                Reduce('b c (x dx) (y dy) -> b c x y', 'max', dx = 2, dy = 2), # 10, 143²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 142²
            ),
            nn.Sequential( # 10, 206²
                Reduce('b c (x dx) (y dy) -> b c x y', 'max', dx = 2, dy = 2), # 10, 71²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 70²
            ),
            nn.Sequential( # 10, 102²
                Reduce('b c (x dx) (y dy) -> b c x y', 'max', dx=2,dy=2), # 10, 35²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 34²
            ),
            nn.Sequential( # 10, 50²
                Reduce('b c (x dx) (y dy) -> b c x y', 'max', dx=2,dy=2), # 10, 17²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 16²
            ),
            nn.Sequential( # 10, 10,  101
                Reduce('b c (x dx) (y dy) -> b c x y', 'max', dx=2,dy=2), # 10, 8²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 7²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 6²
            ),
            nn.Sequential( # 10, 10,  10
                Reduce('b c (x dx) (y dy) -> b c x y', 'max', dx=2,dy=2), # 10, 3²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 2²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 1²
            ),
        ], [
            nn.Sequential( # 10, 1, 1
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 2²
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 3²
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 2, dy = 2), # 10, 6²
            ),
            nn.Sequential( # 10, 1, 1
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 7²
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 8²
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 2, dy = 2), # 10, 16²
            ),
            nn.Sequential( # 10, 1, 1
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 17²
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 2, dy = 2), # 10, 34²
            ),
            nn.Sequential( # 10, 1, 1
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 35²
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 2, dy = 2), # 10, 70²
            ),
            nn.Sequential( # 10, 1, 1
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 71²
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 2, dy = 2), # 10, 142²
            ),
            nn.Sequential( # 10, 1, 1
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 143²
                Repeat('b c x y -> b c (x dx) (y dy)', dx = 2, dy = 2), # 10, 286²
            ),
            nn.Sequential( # 10, 1, 1
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 287²
                ConvTranspose2d(10, 1, (2,2)), nn.Sigmoid(), # 1, 288²
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

model = UNetSpace("unet_space", {
    'views_h' : 9,
    'views_w' : 9,
    'divider_block_h' : 32,
    'divider_block_w' : 32
})
model.eval()
zeros = torch.zeros(1, 1, 9, 9, 64, 64)
zeros_t = torch.zeros(1, 1, 9, 9, 32, 32)
lossf = nn.MSELoss()
with torch.no_grad():
    x = model(zeros)
    print(x.shape)
    print(lossf(zeros_t, x))

