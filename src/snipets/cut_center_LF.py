import os

import numpy as np
from PIL import Image as IMG
import einops


def cut_center_LF(lf, size):
    views = lf.shape[1:3]
    half = lambda x: x // 2
    views_center = tuple(half(dim) for dim in views)
    views_start = tuple(viewdim - (sizedim // 2) for viewdim, sizedim in zip(views_center, size))
    views_end = tuple(viewdim + (sizedim // 2) + (sizedim % 2) for viewdim, sizedim in zip(views_center, size))

    return lf[:, views_start[0]:views_end[0], views_start[1]:views_end[1], :, :]

from PIL import Image as im
def cut_center(img):

    lf = img
    lf = np.array(lf)
    lf = einops.rearrange(lf, '(v h) (u w) c -> c u v h w', u=13, v=13)
    print(f"lf.shape = {lf.shape}")
    lf2 = cut_center_LF(lf, (9, 9))
    print(f"lf2.shape = {lf2.shape}")
    assert ((lf2 == lf[:, 2:-2, 2:-2, :, :]).all())

    lf2 = einops.rearrange(lf2, 'c u v h w -> (v h) (u w) c',  u=9, v=9)


    data = im.fromarray(lf2)
    return data
    # data.save('/home/idm/cut.png')
path='/home/machado/MultiView_RGB/'
pathOut='/home/machado/MultiView_9x9_RGB/'

for classe in (os.listdir(path)):
    os.makedirs(os.path.join(pathOut, classe), exist_ok=True)
    for lf in os.listdir(os.path.join(path,classe)):
        lf_path= os.path.join(path,classe,lf)
        img = (IMG.open(lf_path))
        cuted_lf = cut_center(img)
        cuted_lf.save(os.path.join(pathOut,classe,lf))
