from torch.nn import Conv2d, ConvTranspose2d, Linear
from einops.layers.torch import Rearrange, Reduce
import torch
import torch.nn as nn
import os.path
from unetlike import UNetLike, preserving_dimensions, Repeat
class UNetSpace(nn.Module):
    def __init__(self, name):
        super().__init__()
        s, t, u, v = (13,13,32,32)
        blocker = Rearrange('b c s t (Du u) (Dv v) -> b (Du Dv c) s t u v', u=u,v=v)
        flatener = Rearrange('b c s t u v -> (b s t) c u v', s = s, t = t)
        deflatener = Rearrange('(b s t) c u v -> b c s t u v', s = s, t = t)
        flat_model = UNetLike([ # b*13*13, 6, 32,32 (stuv)
            nn.Sequential(
                Conv2d(6, 10, kernel_size=(3,3)), nn.PReLU(), # b*13*13,10, 30, 30(stuv)
            ),
            nn.Sequential(# b*13*13,10, 30, 30(stuv)
                Rearrange('(b s t) c u v -> (b u v) c s t', s = 13, t = 13, u = 30, v = 30), # b*30*30, 10, 13, 13 (uvst)
                Conv2d(10, 10, kernel_size=(2,2)), nn.PReLU(), # b*30*30, 10, 12, 12(uvst)
                #Rearrange('(b s t) c u v -> (b u v) c s t', s = 30, t = 30, u = 12, v = 12), # b*12*12, 10, 30, 30
            ),
            nn.Sequential( # b*30*30, 10, 12, 12(uvst)
                Reduce('(b u du v dv) c (t dt) (s ds) -> (b u v) c s t', 'max', u=10, du=3, v=10, dv=3, ds = 2, dt=2, s = 6, t=6),
                # b*10*10, 10, 6, 6(uvst)
                Conv2d(10, 10, (3,3)), nn.PReLU(), # b*10*10, 10, 4, 4(uvst)
                #preserving_dimensions(Conv2d, 10, 10), nn.Dropout(), nn.PReLU()
            ),
            nn.Sequential( # b*10*10, 10, 4, 4(uvst)
                Reduce('(b u du v dv) c s t -> (b t s) (c u v)', 'max', u = 5, du=2, v = 5, dv=2, s = 4, t = 4), # b*4*4, 10*5*5(stuv)
                Linear(10*5*5, 10), nn.PReLU(), # b*4*4, 10*1*1(stuv)
                #preserving_dimensions(Conv2d, 10, 10), nn.Dropout(), nn.PReLU()
            ),
            nn.Sequential( # b*4*4, 10*1*1
                Rearrange('(b s t) (c u v) -> (b u v) (c s t)', u = 1, v = 1, s = 4, t = 4), # b*1*1, 10*4*4(uvst)
                Linear(10*4*4, 10), nn.PReLU(), # b*1*1, 10*1*1 (uvst)
                #preserving_dimensions(Conv2d, 10, 10), nn.Dropout(), nn.PReLU()
            ),
        ], [
            nn.Sequential( # b*1*1, 10*1*1 (uvst)
                Linear(10, 10*4*4), nn.PReLU(),# b*1*1, 10*4*4 (uvst)
                Rearrange('(b u v) (c s t)-> (b s t) (c u v)',u=1,v=1,s=4,t=4), # b*4*4, 10, 1, 1(stuv)
            ),
            nn.Sequential( # b*4*4, 10*1*1 (stuv)
                Linear(10, 10*5*5), nn.PReLU(),# b*4*4, 10*5*5 (stuv)
                Repeat('(b s t) (c u v) -> (b u du v dv) c s t',u=5,v=5,du=2,dv=2,s=4,t=4), # b*5*5, 10, 4, 4(uvst)
            ),
            nn.Sequential( # b*10*10, 10, 4,4 (uvst)
                ConvTranspose2d(10, 10, (3,3)), nn.PReLU(), # b*10*10, 10, 6, 6(uvst)
                Repeat('(b u v) c s t -> (b u du v dv) c (t dt) (s ds)', u=10, du=3, v=10, dv=3, ds = 2, dt=2, s = 6, t=6),
                # b*30*30, 10, 12, 12(uvst)
            ),
            nn.Sequential(# b*30*30, 10, 12, 12(uvst)
                ConvTranspose2d(10, 10, kernel_size=(2,2)), nn.PReLU(),  # b*30*30, 10, 13, 13 (uvst)
                Rearrange('(b u v) c s t -> (b s t) c u v', s = 13, t = 13, u = 30, v = 30),# b*13*13,10, 30, 30(stuv)
            ),
            nn.Sequential(
                ConvTranspose2d(10, 1, kernel_size=(3,3)), nn.PReLU(), # b*13*13, 1, 32,32 (stuv)
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

