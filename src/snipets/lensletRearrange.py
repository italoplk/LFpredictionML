import os

import einops as ein

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# normalizer_factor = 2/(2 ** 16 - 1)
#
# img = cv2.imread("/home/idm/Divided.png", cv2.IMREAD_UNCHANGED)
# # img_ycbcr = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB))
# # img_ycbcr = np.array(cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB))
# # img_normalizada = img_ycbcr.astype(np.float32) * 255
#
# # img_normalizada = img_ycbcr.astype(np.float32) * normalizer_factor
#
# # funciona
# # img = np.load('/home/idm/vp_stack1.png.npy', allow_pickle=False)
# lenslet = ein.rearrange(img,' w h c ->  w h c')
#
# plt.imsave("/home/idm/savetest.png", lenslet)

# plt.figure()
# plt.imshow(img_ycbcr, interpolation='none')
# plt.grid(False)
# plt.title('lenslet')
# plt.show()


# img = Image.open("/home/idm/nonDivided.png")


def multiview2lenslet(img, path_rgb, path_gscale, lf_name):
    image_array = np.array(img)
    image_array = ein.rearrange(image_array, '(v h) (u w)  c -> (h v) (w u)  c', u=8, v=8)
    # Convert the NumPy array back to an image using Pillow
    reconstructed_image = Image.fromarray(image_array)

    grayscale_image = reconstructed_image.convert("L")
    grayscale_image.save(os.path.join(path_gscale, lf_name))

    reconstructed_image.save(os.path.join(path_rgb, lf_name))

    # reconstructed_image.show()
    # grayscale_image.show()
    # return reconstructed_image
    # To save the image to a new file
    # reconstructed_image.save("pillowTest.png")

path='/home/machado/MultiView_8x8_RGB/'
pathOut='/home/machado/Lenslet_8x8_RGB/'
pathOutg='/home/machado/Lenslet_8x8_Gscale/'

for classe in os.listdir(path):
    os.makedirs(os.path.join(pathOut, classe), exist_ok=True)
    os.makedirs(os.path.join(pathOutg, classe), exist_ok=True)
    inner_path_rgb = os.path.join(pathOut, classe)
    inner_path_g = os.path.join(pathOutg, classe)
    for lf in os.listdir(os.path.join(path, classe)):
        lf_path = os.path.join(path, classe, lf)
        img = (Image.open(lf_path))
        multiview2lenslet(img, inner_path_rgb, inner_path_g, lf)