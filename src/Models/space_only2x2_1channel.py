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
        flat_model = UNetLike([ # 18, 576²
            nn.Sequential(
                Conv2d(1, 10, (2,2)), nn.PReLU(), # 10, 576²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 575²
            ),
            nn.Sequential( # 10, 574²
                Conv2d(10, 10, (2,2), stride=(2,2)), nn.PReLU(), # 10, 287²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 286²
            ),
            nn.Sequential( # 10, 286²
                Conv2d(10, 10, (2,2), stride=(2,2)), nn.PReLU(), # 10, 143²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 142²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 141²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 140²
            ),
            nn.Sequential( # 10, 140²
                Conv2d(10, 10, (2,2), stride=(2,2)), nn.PReLU(), # 10, 70²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 69²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 68²
            ),
            nn.Sequential( # 10, 68²
                Conv2d(10, 10, (2,2), stride=(2,2)), nn.PReLU(), # 10, 34²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 33²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 32²
            ),
            nn.Sequential( # 10, 32²
                Conv2d(10, 10, (2,2), stride=(2,2)), nn.PReLU(), # 10, 16²
                Conv2d(10, 10, (2,2)), nn.PReLU(), # 10, 15²
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
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 16²
                ConvTranspose2d(10, 10, (2,2), stride=(2,2)), nn.PReLU(), # 10, 32²
            ),
            nn.Sequential( # 10, 32²
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 33²
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 34²
                ConvTranspose2d(10, 10, (2,2), stride=(2,2)), nn.PReLU(), # 10, 68²
            ),
            nn.Sequential( # 10, 68²
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 69²
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 70²
                ConvTranspose2d(10, 10, (2,2), stride=(2,2)), nn.PReLU(), # 10, 140²
            ),
            nn.Sequential( # 10, 140²
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 141²
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 142²
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 143²
                ConvTranspose2d(10, 10, (2,2), stride=(2,2)), nn.PReLU(), # 10, 286²
            ),
            nn.Sequential( # 10, 286²
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 287²
                ConvTranspose2d(10, 10, (2,2), stride=(2,2)), nn.PReLU(), # 10, 574²
            ),
            nn.Sequential( # 10, 574²
                ConvTranspose2d(10, 10, (2,2)), nn.PReLU(), # 10, 575²
                ConvTranspose2d(10, 1, (2,2)), nn.Sigmoid(), # 1, 576²
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
        return self.f(X)
params = Namespace()
dims = (9,9,64,64)
dims_out = (9,9,32,32)
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

