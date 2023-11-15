import os

import numpy as np
from PIL import Image


# Create 10 random floats in range 0..1 in array "b"

# Read back into different array "r"
# y_cb_cr_array = np.load('/home/idm/vp_s2ack1.png.npy')
# im = Image.fromarray(r.astype(np.uint16))
# im.save("t1.jpeg")
#
# r = np.load('vp_s2ack1.png.npy')
# im = Image.fromarray(r.astype(np.uint8))
# im.save("t2.jpeg")

def saveNPYasPNG(img_array, path, lf_name):

    # y = img_array[:, :, 0]
    # cb = img_array[:, :, 1]
    # cr = img_array[:, :, 2]
    #
    # r = y  * 255.0
    # g = cb * 255.0
    # b = cr * 255.0
    rgb_array = img_array*255.0
    # Stack the RGB channels to create an RGB image
    # print("red: ", r.max())
    # print("green: ", g.max())
    # print("blue: ", b.max())
    # rgb_array = np.stack((r, g, b), axis=-1)

    # Create a PIL Image from the RGB array
    image = Image.fromarray(rgb_array.astype(np.uint8))

    # Save the image to a PNG file
    image.save(os.path.join(path, lf_name))
    return image

# array = np.load("/home/idm/vp_stack1.png.npy")
# saveNPYasPNG(array,"/home/idm", "printMax.png" )

# width, height = y.shape[1], y.shape[0]
# filename = "ycbcr_image.ppm"
#
# with open(filename, "wb") as f:
#     f.write(f"P6 {width} {height} 255\n".encode())
#     y_bytes = (y * 255).astype(np.uint8).tobytes()
#     cb_bytes = (cb * 255).astype(np.uint8).tobytes()
#     cr_bytes = (cr * 255).astype(np.uint8).tobytes()
#     f.write(y_bytes + cr_bytes)
#
# print(f"YCbCr image saved as '{filename}'.")