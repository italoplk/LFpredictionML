from torch.nn import Conv2d, ConvTranspose2d
from einops.layers.torch import Rearrange, Reduce
import torch
import torch.nn as nn
import os.path

#from torchsummary import summary

from unetlike import UNetLike, preserving_dimensions, Repeat


class UNetSpace(nn.Module):
    def __init__(self, name):
        super().__init__()
        u, v = (13, 13)
        blocker = Rearrange('b c u v (bs s) (bt t) -> b (bs bt c) u v s t', s=32, t=32)
        flatener = Rearrange('b c u v s t -> b c (u s) (v t)', u=u, v=v)
        deflatener = Rearrange('b c (u s) (v t) -> b c u v s t', u=u, v=v)
        flat_model = UNetLike([
            nn.Sequential(#10 chanels arbitrary
                preserving_dimensions(Conv2d, 6, 10), nn.PReLU()  # 10, 416, 416
            ),
            nn.Sequential(
                Conv2d(10, 10, (4,4), 4),  nn.PReLU(),  # 10, 208, 208
                # preserving_dimensions(Conv2d, 10, 10), nn.Dropout(), nn.PReLU()
            ),
            nn.Sequential(

                Conv2d(10, 10, (4, 4), 2), nn.PReLU(),  # 10, 104, 104

                # preserving_dimensions(Conv2d, 10, 10), nn.Dropout(), nn.PReLU()
            ),
            nn.Sequential(
                Conv2d(10, 10, (4, 4), 4),  nn.PReLU(),  # 10, 52, 52
            ),

        ], [

            nn.Sequential(

                Repeat('b c x y -> b c (x dx) (y dy)', dx=2, dy=2),  # 10, 104, 104
                ConvTranspose2d(10, 10, (4, 4)), nn.PReLU(),



            ),
            nn.Sequential(  # 10, 40, 40

                ConvTranspose2d(10, 10, (4, 4)), nn.PReLU(),  #
                Repeat('b c x y -> b c (x dx) (y dy)', dx=4, dy=4),  # 10, 208, 208


            ),
            nn.Sequential(  # 10, 204, 204

                Repeat('b c x y -> b c (x dx) (y dy)', dx=4, dy=4),  # 10, 416, 416


                 preserving_dimensions(Conv2d, 10, 10), nn.PReLU(),
            ),
            nn.Sequential(
                preserving_dimensions(Conv2d, 10, 1)
            )
        ], compose=lambda x, y: x + y)
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
        flipped = map(flip, (d, u, l), ((-2, -1), (-2,), (-1,)))
        X = torch.cat((X, *flipped), dim=1)
        # print(X.shape)
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
    #print(summary(model, zeros))

