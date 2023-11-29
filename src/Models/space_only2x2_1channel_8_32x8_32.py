from argparse import Namespace
from torch.nn import Conv2d, ConvTranspose2d
from einops.layers.torch import Rearrange, Reduce
import torch
import torch.nn as nn
import os.path
from Models.unetlike import UNetLike, preserving_dimensions, Repeat
class UNetSpace(nn.Module):
    def __init__(self, name, params):
        super().__init__()
        s, t, u, v = (params.num_views_ver, params.num_views_hor, params.predictor_size, params.predictor_size)
        flat_model = UNetLike([ # 18, 512²
            nn.Sequential(
                Conv2d(1, 10, (2,2)), nn.PReLU(), # 10, 511²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 510²
            ),
            nn.Sequential( # 10, 510²
                Conv2d(10, 10, (2,2), stride=(2,2)), nn.PReLU(), # 10, 255²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 254²
            ),
            nn.Sequential( # 10, 254²
                Conv2d(10, 10, (2,2), stride=(2,2)), nn.PReLU(), # 10, 127²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 126²
            ),
            nn.Sequential( # 10, 126²
                Conv2d(10, 10, (2,2), stride=(2,2)), nn.PReLU(), # 10, 63²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 62²
            ),
            nn.Sequential( # 10, 62²
                Conv2d(10, 10, (2,2), stride=(2,2)), nn.PReLU(), # 10, 31²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 30²
            ),
            nn.Sequential( # 10, 30²
                Conv2d(10, 10, (2,2), stride=(2,2)), nn.PReLU(), # 10, 15²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 14²
            ),
            nn.Sequential( # 10, 14²
                Conv2d(10, 10, (2,2), stride=(2,2)), nn.PReLU(), # 10, 7²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 6²
            ),
            nn.Sequential( # 10, 6²
                Conv2d(10, 10, (2,2), stride=(2,2)), nn.PReLU(), # 10, 3²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 2²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 1²
            ),
        ], [
            nn.Sequential( # 10, 1²
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 2²
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 3²
                ConvTranspose2d(10, 10, (2,2), stride=(2,2)), nn.PReLU(), # 10, 6²
            ),
            nn.Sequential( # 10, 6²
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 7²
                ConvTranspose2d(10, 10, (2,2), stride=(2,2)), nn.PReLU(), # 10, 14²
            ),
            nn.Sequential( # 10, 14²
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 15²
                ConvTranspose2d(10, 10, (2,2), stride=(2,2)), nn.PReLU(), # 10, 30²
            ),
            nn.Sequential( # 10, 30²
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 31²
                ConvTranspose2d(10, 10, (2,2), stride=(2,2)), nn.PReLU(), # 10, 62²
            ),
            nn.Sequential( # 10, 62²
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 63²
                ConvTranspose2d(10, 10, (2,2), stride=(2,2)), nn.PReLU(), # 10, 126²
            ),
            nn.Sequential( # 10, 126²
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 127²
                ConvTranspose2d(10, 10, (2,2), stride=(2,2)), nn.PReLU(), # 10, 254²
            ),
            nn.Sequential( # 10, 254²
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 255²
                ConvTranspose2d(10, 10, (2,2), stride=(2,2)), nn.PReLU(), # 10, 510²
            ),
            nn.Sequential( # 10, 510²
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 511²
                ConvTranspose2d(10, 1, (2,2)), nn.Sigmoid(), # 1, 512²
            ),
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
        #assert(tuple(X.shape[1:]) == (1,8*64,8*64))
        return self.f(X)
params = Namespace()
dims = (8,8,64,64)
dims_out = (8,8,32,32)
(params.num_views_ver, params.num_views_hor, params.predictor_size, params.predictor_size) = dims_out

model = UNetSpace("unet_space", params)
model.eval()
zeros = torch.zeros(1, 1, dims[0]*dims[2], dims[1]*dims[3])
zeros_t = torch.zeros(1, 1, dims_out[0]*dims_out[2], dims_out[1]*dims_out[3])
lossf = nn.MSELoss()
with torch.no_grad():
    x = model(zeros)
    x = x[:,:,-dims_out[0]*dims_out[2]:,-dims_out[1]*dims_out[3]:]
    print(x.shape)
    print(lossf(zeros_t, x))

