from einops import reduce
from dataset_reader import read_LF
import os

decoded = "/home/machado/Decoded_LFs/png/decoded_32_noPartition/"
original = "/home/machado/Original_LFs/png/"



for folder in os.listdir(decoded):
    for lf in os.listdir(os.path.join(decoded,folder)):
        #if "75" in lf:
        path_decoded = os.path.join(decoded, folder, lf.split('.png')[0], 'bpp_0.75.png')
        lf_decoded = read_LF(path_decoded)



        # path_original = os.path.join(original, folder, lf+'.mat.png')
        # lf_original = read_LF(path_original)

