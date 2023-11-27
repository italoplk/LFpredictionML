import Params
import numpy as np
import cv2
from einops import rearrange
import os


class LightField:

    def __init__(self, lf_path, lf_name):
        self.name = lf_name
        self.path = lf_path
        self.classname = lf_path.split("/")[-1]

        self.full_path = os.path.join(lf_path, lf_name)

    def __str__(self):
        return ', '.join([self.name, self.path, self.full_path])


    normalizer_factor_16bit = 2 / ((2 ** 16) - 1)
    normalizer_factor_8bit = 2 / ((2 ** 8) - 1)

    @classmethod
    def normalize_image(cls, image, bit_depth: int):

        if bit_depth == 16:
            return image.astype(np.float32) * cls.normalizer_factor_16bit-1
        elif bit_depth == 8:
            return image.astype(np.float32) * cls.normalizer_factor_8bit-1
        else:
            print("Image type not supported, implementation necessary.")
            exit(255)

    @classmethod
    def denormalize_image(cls, image, bit: int, is_prelu: bool):
        if bit == 8:
            return ((image + is_prelu) / cls.normalizer_factor_8bit).astype(np.uint8)
        elif bit == 16:
            return ((image + is_prelu) / cls.normalizer_factor_16bit).astype(np.uint16)


    # write the LFs after validation.
    @classmethod#color can be L (luma) or RGB (gscale rgb)
    def write_LF_PNG(cls, image: np.uint8, path: str, nviews_ver: int, nviews_hor: int, nbits: int,
                     color: str = 'L'):
        try:  # @TODO check ver and hor orders E np.uint8 image
            image = rearrange(image, 'c s t u v -> (s u) (t v) c', s=nviews_ver, t=nviews_hor)

            # In THEORY, shape[-1] = 3 é RGB e =1 é Gscale == luma
            image = cls.denormalize_image(image, image.itemsize * 8, image.shape[-1])
            cv2.imwrite(f'{path}.png', image)

        except RuntimeError as e:
            print("Failed to save LF: ", e.__traceback__)


    # @TODO assumir que todo LF vai entrar previamente arranjado de acordo com o modelo
    def load_lf(self):
        try:
            img = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
        except RuntimeError as e:
            print("Failed to open image path: ", e.__traceback__)
            exit()

        # color conversion excludes other color channels
        #@TODO supor que vai entrar em luma
        img_luma = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB))
        #return the normalized image
        return self.normalize_image(img_luma, img_luma.itemsize * 8)


