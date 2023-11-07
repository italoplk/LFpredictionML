import torch

def semi_flatten(x):
    # x: [1,1,13,13,64,64] already random cropped
    x = x.squeeze(0).squeeze(0)
    S,U,H,W = x.shape
    x = x.reshape(S*U,H,W)
    y = x[:,32:,32:].clone() # S*U, 32, 32
    x[:,32:,32:] = 0 # S*U, 64, 64
    return x, y

def full_flatten(x):
    # x: [1,1,13,13,64,64] already random cropped
    x,y = semi_flatten(x)
    # x: [13*13,64,64]
    # y: [13*13,32,32]
    a = x[:,:32,:32].clone().reshape(1,416,416)
    b = x[:,32:,:32].clone().reshape(1,416,416)
    c = x[:,:32,32:].clone().reshape(1,416,416)
    d = x[:,32:,32:].clone().reshape(1,416,416)
    y = y.reshape(1,416,416)
    x = torch.empty((1,416*2,416*2)).to(x.device)
    x[:, :416, :416] = a
    x[:, 416:, :416] = b
    x[:, :416, 416:] = c
    x[:, 416:, 416:] = d

    return x,y

if __name__ == '__main__':
    device = "cuda"
    x = torch.rand((1,1,13,13,64,64)).to(device)
    x,y = semi_flatten(x)
    print(x.shape)
    print(y.shape)
    print(x.device)
    print(y.device)

    #assert torch.sum(x[:,416:, 416:] == 0)
