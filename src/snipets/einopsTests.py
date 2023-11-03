import einops as ein
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image



normalizer_factor = 2/(2 ** 16 - 1)

img = cv2.imread("/home/idm/Divided.png", cv2.IMREAD_UNCHANGED)
img_ycbcr = np.array(cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb))
# img_ycbcr = np.array(cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB))
img_normalizada = img_ycbcr.astype(np.float32) * 255

# funciona
# img = np.load('/home/idm/vp_stack1.png.npy', allow_pickle=False)
lenslet = ein.rearrange(img_ycbcr,' w h c ->  w h c')

plt.imsave("/home/idm/savetest.png", lenslet)
plt.figure()
plt.imshow(img_ycbcr, interpolation='none')
plt.grid(False)
plt.title('lenslet')
plt.show()