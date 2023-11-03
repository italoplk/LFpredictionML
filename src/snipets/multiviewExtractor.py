import sys


try:
    import plenopticam as pcam
except ImportError:
    import plenopticam as pcam
print('PlenoptiCam v'+pcam.__version__+'\n')

try:
    import matplotlib.pyplot as plt
except:
    import matplotlib.pyplot as plt


import tifffile as tiff

from PIL import Image as im
import numpy as np

from matplotlib import animation
from IPython.display import HTML


# loader = pcam.misc.DataDownloader()
# loader.download_data(loader.host_eu_url, fp='./data')
# loader.extract_archive(archive_fn='./data/illum_test_data.zip', fname_list='lfr')

# instantiate config object and set image file paths and options
cfg = pcam.cfg.PlenopticamConfig()
cfg.default_values()
cfg.params[cfg.lfp_path] = '/home/idm/Downloads/People/Fountain_&_Vincent_2.LFR'
cfg.params[cfg.cal_path] = '/home/idm/Downloads/caldata-B5143909630.tar'
# cfg.params[cfg.lfp_path] = '/scratch/Fountain_Vincent_2.LFR'
# cfg.params[cfg.cal_path] = '/scratch/caldata-B5143909630.tar'

cfg.params[cfg.opt_cali] = True
cfg.params[cfg.ptc_leng] = 13
cfg.params[cfg.cal_meth] = pcam.cfg.constants.CALI_METH[3]

# instantiate status object for progress
sta = pcam.misc.PlenopticamStatus()

reader = pcam.lfp_reader.LfpReader(cfg, sta)
# reader.main()
reader.main("png")
lfp_img = reader.lfp_img

# plt.figure()
# plt.imshow(lfp_img, cmap='gray', interpolation='none')
# plt.grid(False)
# plt.title('Raw Illum image')
# plt.show()
#
#finds proper calibration on the tar
cal_finder = pcam.lfp_calibrator.CaliFinder(cfg, sta)
ret = cal_finder.main()
wht_img = cal_finder.wht_bay

#performs calibration
cal_obj = pcam.lfp_calibrator.LfpCalibrator(wht_img, cfg, sta)
ret = cal_obj.main()
cfg = cal_obj.cfg

import matplotlib as mpl
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['pdf.fonttype'] = 42

y_coords = [row[0] for row in cfg.calibs[cfg.mic_list]]
x_coords = [row[1] for row in cfg.calibs[cfg.mic_list]]

s = 3
h, w, c = wht_img.shape if len(wht_img.shape) == 3 else wht_img.shape + (1,)
hp, wp = [39]*2
fig, axs = plt.subplots(s, s, facecolor='w', edgecolor='k', figsize=(9, 9))
markers = ['r.', 'b+', 'gx']
labels = [r'$\bar{\mathbf{c}}_{j,h}$',
          r'$\tilde{\mathbf{c}}_{j,h}$ with $\beta=0$',
          r'$\tilde{\mathbf{c}}_{j,h}$ with $\beta=1$']
m = 2

for i in range(s):
    for j in range(s):
        # plot cropped image part
        k = h//2 + (i-s//2) * int(h/2.05) - hp // 2
        l = w//2 + (j-s//2) * int(w/2.05) - wp // 2
        axs[i, j].imshow(wht_img[k:k+hp, l:l+wp, ...], cmap='gray')

        # plot centroids in cropped area
        coords_crop = [(y, x) for y, x in zip(y_coords, x_coords)
                       if k <= y <= k+hp-.5 and l <= x <= l+wp-.5]
        y_centroids = [row[0] - k for row in coords_crop]
        x_centroids = [row[1] - l for row in coords_crop]
        axs[i, j].plot(x_centroids, y_centroids, markers[m],
                       markersize=10, label=labels[m])
        axs[i, j].grid(False)

        if j == 0 or i == s-1:
            if j == 0 and i == s-1:
                axs[i, j].tick_params(top=False, bottom=True, left=True, right=False,
                                      labelleft=True, labelbottom=True)
                axs[i, j].set_yticklabels([str(k), str(k+hp//2), str(k+hp)])
                axs[i, j].set_xticklabels([str(l), str(l+wp//2), str(l+wp)])
            elif j == 0:
                axs[i, j].tick_params(top=False, bottom=True, left=True, right=False,
                                      labelleft=True, labelbottom=False)
                axs[i, j].set_yticklabels([str(k), str(k+hp//2), str(k+hp)])
            elif i == s-1:
                axs[i, j].tick_params(top=False, bottom=True, left=True, right=False,
                                      labelleft=False, labelbottom=True)
                axs[i, j].set_xticklabels([str(l), str(l+wp//2), str(l+wp)])

        else:
            axs[i, j].tick_params(top=False, bottom=True, left=True, right=False,
                                  labelleft=False, labelbottom=False)

        axs[i, j].set_yticks(range(0, hp+1, hp//2))
        axs[i, j].set_xticks(range(0, wp+1, wp//2))


#micro image alignment
ret = cfg.load_cal_data()
aligner = pcam.lfp_aligner.LfpAligner(lfp_img, cfg, sta, wht_img)
ret = aligner.main()
lfp_img_align = aligner.lfp_img

from os.path import join, basename
import pickle

with open(join(cfg.exp_path, 'lfp_img_align.pkl'), 'rb') as f:
    lfp_img_align = pickle.load(f)
#
# plt.figure()
# plt.imshow(lfp_img_align / lfp_img_align.max(), interpolation='none')
# plt.grid(False)
# plt.title('Aligned Illum image')
# plt.show()

try:
    from plenopticam.lfp_reader import LfpDecoder

    # try to load json file (if present)
    json_dict = cfg.load_json(cfg.exp_path, basename(cfg.exp_path) + '.json')
    cfg.lfpimg = LfpDecoder.filter_lfp_json(json_dict, cfg.lfpimg)
except FileNotFoundError:
    pass


extractor = pcam.lfp_extractor.LfpExtractor(lfp_img_align, cfg, sta)
ret = extractor.main()
vp_img_arr = extractor.vp_img_arr



view_obj = pcam.lfp_extractor.LfpViewpoints(vp_img_arr=vp_img_arr)
vp_stack = view_obj.views_stacked_img

# data = im.fromarray(view_obj)

# saving the final output
# as a PNG file


# np.save('vp_stack1.png', vp_stack)
# np.save('vp_s2ack1.png', vp_stack/vp_stack.max())
plt.figure()
plt.imshow(vp_stack/vp_stack.max(), interpolation='none')
plt.grid(False)
plt.title('All sub-aperture images view')
plt.show()

data = im.fromarray((vp_stack*255).astype(np.uint8))
data.save('/home/idm/Downloads/People/Fountain_&_Vincent_2/try2outPutSubApertureX255.png')
# data.save('/home/machado/scratch/try2outPutSubAperture.png')


# tiff.imwrite("tiffDivided.tiff", vp_stack/vp_stack.max())
# tiff.imwrite("tiffNonDivided.tiff", vp_stack)




