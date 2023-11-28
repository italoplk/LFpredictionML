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
        s, t = (13,13)
        blocker = Rearrange('b c s t (bu u) (bv v) -> b (bu bv c) s t u v', u=32,v=32)
        flatener = Rearrange('b c s t u v -> b c (s u) (t v)', s = s, t = t)
        deflatener = Rearrange('b c (s u) (t v) -> b c s t u v', s = s, t = t)
        flat_model = UNetLike([
            nn.Sequential(#10 chanels arbitrary
                preserving_dimensions(Conv2d, 6, 32),  nn.PReLU()  # 10, 416, 416
            ),

            nn.Sequential(
                Conv2d(32, 64, (2,2), 2),  nn.PReLU(),  # 10, 208, 208
                # preserving_dimensions(Conv2d, 10, 10), nn.Dropout(), nn.PReLU()
            ),

            nn.Sequential(

                Conv2d(64, 128, (4, 4), 4),  nn.PReLU(),  # 10, 52, 52
                #Conv2d(10, 10, (3, 3), 1), nn.Dropout(), nn.PReLU(),  # 10, 24, 24
                # preserving_dimensions(Conv2d, 10, 10), nn.Dropout(), nn.PReLU()
            ),

            nn.Sequential(
                Conv2d(128, 256, (4, 4), 4),  nn.PReLU(),  # 10, 13, 13

            ),

        ], [

            nn.Sequential(

                preserving_dimensions(Conv2d, 256, 128), nn.PReLU(), #13, 13
                Repeat('b c x y -> b c (x dx) (y dy)', dx=4, dy=4),  # 10, 52, 52

            ),
            nn.Sequential(


                preserving_dimensions(Conv2d, 128, 64), nn.PReLU(),
                Repeat('b c x y -> b c (x dx) (y dy)', dx=4, dy=4),  # 10, 208, 208


            ),
            nn.Sequential(

                preserving_dimensions(Conv2d, 64, 32), nn.PReLU(),
                Repeat('b c x y -> b c (x dx) (y dy)', dx=2, dy=2),  # 10, 208, 208

            ),
            nn.Sequential(
                preserving_dimensions(Conv2d, 32, 1) #416, 416
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
    # print(summary(model, zeros))

