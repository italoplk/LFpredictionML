#
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import numpy as np
#
# # Load the PNG image
# image_path = '/home/idm/Original_LFs/Nature/Bush.mat.png'
# img = mpimg.imread(image_path)
#
# # Separate the image into its RGB channels
# red_channel = img[:, :, 1]
# # green_channel = img[:, :, 1]
# # blue_channel = img[:, :, 2]
#
# # Create bin edges based on the 16-bit range (0 to 65535)
# bin_edges = np.linspace(0, 65535, 65536)  # 16-bit RGB
#
# # Create histograms for each channel
# red_hist, _ = np.histogram(red_channel, bins=bin_edges)
# # green_hist, _ = np.histogram(green_channel, bins=bin_edges)
# # blue_hist, _ = np.histogram(blue_channel, bins=bin_edges)
#
# # Plot the histograms for each channel
# plt.figure(figsize=(12, 6))
#
# plt.subplot(131)
# plt.title('Red Channel Histogram')
# plt.xlabel('Pixel Value')
# plt.ylabel('Frequency')
# plt.bar(bin_edges[:-1], red_hist, width=1, color='red')
# plt.xlim(0, 65535)
# #
# # plt.subplot(132)
# # plt.title('Green Channel Histogram')
# # plt.xlabel('Pixel Value')
# # plt.ylabel('Frequency')
# # plt.bar(bin_edges[:-1], green_hist, width=1, color='green')
# # plt.xlim(0, 65535)
# #
# # plt.subplot(133)
# # plt.title('Blue Channel Histogram')
# # plt.xlabel('Pixel Value')
# # plt.ylabel('Frequency')
# # plt.bar(bin_edges[:-1], blue_hist, width=1, color='blue')
# # plt.xlim(0, 65535)
#
# plt.tight_layout()
# plt.show()


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Load the PNG image
# image_path = '/home/idm/Original_LFs/Nature/Bush.mat.png'
image_path = '/home/idm/bpp_0.75.png'
img = mpimg.imread(image_path)

# If the image has color channels, convert it to grayscale for histogram
if img.ndim == 3:
    img = np.mean(img, axis=2)

# Get the minimum and maximum pixel values in the image
min_pixel_value = img.min()
max_pixel_value = img.max()

# Create bin edges based on the minimum and maximum values
bin_edges = np.linspace(min_pixel_value, max_pixel_value, 256)

# Create a histogram of pixel values
hist, _ = np.histogram(img, bins=bin_edges)

# Plot the histogram
plt.figure(figsize=(8, 6))
plt.title('Image Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.bar(bin_edges[:-1], hist, width=(bin_edges[1] - bin_edges[0]))
plt.xlim(min_pixel_value, max_pixel_value)
plt.show()
