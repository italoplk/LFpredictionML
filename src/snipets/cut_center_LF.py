import numpy as np
import cv2
import einops

def cut_center_LF(lf, size):
    views = lf.shape[1:3]
    half = lambda x: x//2
    views_center = tuple(half(dim) for dim in views)
    views_start = tuple(viewdim - (sizedim // 2) for viewdim, sizedim in zip(views_center, size))
    views_end = tuple(viewdim + (sizedim // 2) + (sizedim % 2) for viewdim, sizedim in zip(views_center, size))
    
    return lf[:,views_start[0]:views_end[0],views_start[1]:views_end[1],:,:]

if __name__ == "__main__":
    lf = np.zeros((3, 13,13,10,10))
    print(f"lf.shape = {lf.shape}")
    lf2 = cut_center_LF(lf, (9,9))
    print(f"lf2.shape = {lf2.shape}")
    assert(tuple(lf2.shape) == (3, 9, 9, 10, 10))

    lf = cv2.imread("/scratch/Original_LFs/png/Buildings/Black_Fence.mat.png", cv2.IMREAD_UNCHANGED | cv2.IMREAD_COLOR)
    lf = einops.rearrange(lf, '(v h) (u w) c -> c u v h w', u=13,v=13)
    print(f"lf.shape = {lf.shape}")
    lf2 = cut_center_LF(lf, (9,9))
    print(f"lf2.shape = {lf2.shape}")
    assert(tuple(lf2.shape) == (3, 9, 9, 434, 625))
    assert((lf2 == lf[:,2:-2, 2:-2,:,:]).all())
