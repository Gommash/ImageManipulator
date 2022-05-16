import cv2
import numpy as np

import torchvision.transforms as transforms
from PIL import ImageEnhance, Image, ImageOps
from skimage import restoration, io, color, util, img_as_ubyte
from scipy.signal import convolve2d as conv2


class WrappedImage:
    def __init__(self, src, img_class_type: str):
        self.img_obj = ""
        self.img_class_type = img_class_type
        if img_class_type == "cv2":
            self.img_obj = cv2.imread(src)
            self.img_obj = cv2.cvtColor(self.img_obj, cv2.COLOR_RGB2BGR)
            self.img_obj = cv2.cvtColor(self.img_obj, cv2.COLOR_BGR2GRAY)
        elif img_class_type == "pil":
            self.img_obj = Image.open(src)
            self.img_obj = self.img_obj.convert("RGB")
            self.img_obj = ImageOps.grayscale(self.img_obj)

        elif img_class_type == "skimage":
            self.img_obj = io.imread(src)
            if len(self.img_obj.shape) == 3:
                self.img_obj = color.rgb2gray(self.img_obj)

    def get_img_classtype(self) -> str:
        return self.img_class_type

    def get_img_instance(self):
        return self.img_obj


class ImageFilter(WrappedImage):
    def __init__(self, img_obj, img_class_type: str):
        super().__init__(img_obj, img_class_type)

    def enhance(self):
        pass


class ImageTransform(WrappedImage):
    def __init__(self, src, img_class_type: str):
        super().__init__(src, img_class_type)

    def transform(self):
        pass


# IMAGE FILTERS

# Background Remover
class RollingBallFilter(ImageFilter):
    def __init__(self, src, filepath, radius=100.0, invert=False):
        super().__init__(src, "skimage")
        self.radius = radius
        self.filepath = filepath
        self.invert = invert

    def enhance(self):
        image = util.img_as_float64(self.img_obj)
        gray_image = image

        normalized_radius = self.radius / 255
        kernel = restoration.ellipsoid_kernel(
            (self.radius * 2, self.radius * 2),
            normalized_radius * 2
        )
        background = restoration.rolling_ball(gray_image, kernel=kernel)
        if self.invert:
            background = util.invert(background)
            gray_image = util.invert(gray_image)
            filtered_image = util.invert(gray_image - background)
        else:
            filtered_image = gray_image - background
        img_n = cv2.normalize(src=filtered_image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                              dtype=cv2.CV_8U)
        io.imsave(self.filepath, img_n)
        return 0


# histogram equalization

class CLAHEFilter(ImageFilter):
    def __init__(self, src, cliplimit=2, tilegridsize=(10, 10), factor=0):
        super().__init__(src, "cv2")
        self.cliplimit = cliplimit
        self.tilegridsize = tilegridsize
        self.factor = factor

    def enhance(self):
        return cv2.createCLAHE(clipLimit=self.cliplimit, tileGridSize=self.tilegridsize).apply(
            self.img_obj) + self.factor


class HistogramEqualizationFilter(ImageFilter):
    def __init__(self, src):
        super().__init__(src, "cv2")

    def enhance(self):
        return cv2.equalizeHist(self.img_obj)


# Sharpness
class SharpnessFilter(ImageFilter):
    def __init__(self, src, factor=1.0):
        super().__init__(src, "pil")
        self.factor = factor

    def enhance(self):
        return np.array(ImageEnhance.Sharpness(self.img_obj).enhance(self.factor))


# Brightness
class BrightnessFilter(ImageFilter):
    def __init__(self, src, factor=1.0):
        super().__init__(src, "pil")
        self.factor = factor

    def enhance(self):
        return np.array(ImageEnhance.Brightness(self.img_obj).enhance(self.factor))


# Contrast
class ContrastFilter(ImageFilter):
    def __init__(self, src, factor=1.0):
        super().__init__(src, "pil")
        self.factor = factor

    def enhance(self):
        return np.array(ImageEnhance.Contrast(self.img_obj).enhance(self.factor))


# Color
class ColorFilter(ImageFilter):
    def __init__(self, src, factor=1.0):
        super().__init__(src, "pil")
        self.factor = factor

    def enhance(self):
        return np.array(ImageEnhance.Color(self.img_obj).enhance(self.factor))


# Blur

class GaussianBlurFilter(ImageFilter):
    def __init__(self, src, blurgrid=(10, 10), border=cv2.BORDER_DEFAULT):
        super().__init__(src, "cv2")
        self.blurgrid = blurgrid
        self.border = border

    def enhance(self):
        return cv2.GaussianBlur(self.img_obj, self.blurgrid, self.border)


# Deblur / Denoise
class RichardsonLucyFilter(ImageFilter):
    def __init__(self, src, num_iter=100):
        super().__init__(src, "skimage")
        self.psf = np.ones((5, 5)) / 25
        self.num_iter = num_iter
        self.img_obj = conv2(self.img_obj, self.psf, 'same')

    def enhance(self):
        return restoration.richardson_lucy(self.img_obj, psf=self.psf
                                           , num_iter=self.num_iter)


# Denoise
class DenoiseWienerFilter(ImageFilter):
    def __init__(self, src, balance=1100):
        super().__init__(src, "skimage")
        self.psf = np.ones((5, 5)) / 25
        self.img_obj = conv2(color.rgb2gray(self.img_obj), self.psf, 'same')
        self.balance = balance

    def enhance(self):
        return restoration.wiener(self.img_obj, psf=self.psf, balance=self.balance)


class DenoiseNLMeansFilter(ImageFilter):
    def __init__(self, src, patch_size=7, patch_distance=5, h=0.2, fast_mode=False):
        super().__init__(src, "skimage")
        self.patch_size = patch_size
        self.patch_distance = patch_distance
        self.h = h
        self.fast_mode = fast_mode

    def enhance(self):
        return restoration.denoise_nl_means(self.img_obj, self.patch_size, self.patch_distance, self.h,
                                            fast_mode=self.fast_mode)


class DenoiseBilateralFilter(ImageFilter):
    def __init__(self, src, sigma_color=0.05, sigma_spatial=15):
        super().__init__(src, "skimage")
        self.sigma_color = sigma_color
        self.sigma_spatial = sigma_spatial

    def enhance(self):
        return restoration.denoise_bilateral(self.img_obj, sigma_color=self.sigma_color,
                                             sigma_spatial=self.sigma_spatial)


class DenoiseTvBregmanFilter(ImageFilter):
    def __init__(self, src, weight=5.0, max_num_iter=100, isotropic=True):
        super().__init__(src, "skimage")
        self.weight = weight
        self.max_num_iter = max_num_iter
        self.isotropic = isotropic

    def enhance(self):
        return restoration.denoise_tv_bregman(self.img_obj, weight=self.weight,
                                              max_num_iter=self.max_num_iter, isotropic=self.isotropic)


class DenoiseTvChambolleFilter(ImageFilter):
    def __init__(self, src, weight=0.1, max_num_iter=200, isotropic=True):
        super().__init__(src, "skimage")
        self.weight = weight
        self.max_num_iter = max_num_iter
        self.isotropic = isotropic

    def enhance(self):
        return restoration.denoise_tv_chambolle(self.img_obj, weight=self.weight,
                                                max_num_iter=self.max_num_iter)


class DenoiseWaveletFilter(ImageFilter):
    def __init__(self, src, sigma=0.1, rescale_sigma=True):
        super().__init__(src, "skimage")
        self.sigma = sigma
        self.rescale_sigma = rescale_sigma

    def enhance(self):
        return restoration.denoise_wavelet(self.img_obj, sigma=self.sigma
                                           , rescale_sigma=True)


# Edge detection
class SobelEdgeDetectionFilter(ImageFilter):
    def __init__(self, src, depth=cv2.CV_16S, delta=0, scale=1.0, ksize=3, xaxis=True):
        super().__init__(src, "cv2")
        self.depth = depth
        self.delta = delta
        self.scale = scale
        self.ksize = ksize
        self.img_obj = GaussianBlurFilter(src, blurgrid=(self.ksize, self.ksize)).enhance()
        self.xaxis = xaxis

    def enhance(self):
        if self.xaxis:
            return cv2.Sobel(self.img_obj, self.depth, 1, 0, ksize=self.ksize, scale=self.scale, delta=self.delta,
                             borderType=cv2.BORDER_DEFAULT)
        else:
            return cv2.Sobel(self.img_obj, self.depth, 0, 1, ksize=self.ksize, scale=self.scale, delta=self.delta,
                             borderType=cv2.BORDER_DEFAULT)


class LaplacianEdgeDetectionFilter(ImageFilter):
    def __init__(self, src, depth=cv2.CV_16S, ksize=3, delta=0, scale=1):
        super().__init__(src, "cv2")
        self.depth = depth
        self.ksize = ksize
        self.delta = delta
        self.scale = scale
        self.img_obj = GaussianBlurFilter(src, blurgrid=(3, 3)).enhance()

    def enhance(self):
        return cv2.Laplacian(self.img_obj, ksize=self.ksize, delta=self.delta, scale=self.scale, ddepth=self.depth)


class CannyEdgeDetectionFilter(ImageFilter):
    def __init__(self, src: str, threshold=(100, 100), aperture_size=5, l2gradient=True):
        super().__init__(src, "cv2")
        self.threshold = threshold
        self.aperture_size = aperture_size
        self.L2Gradient = l2gradient
        self.img_obj = GaussianBlurFilter(src, blurgrid=(3, 3)).enhance()

    def enhance(self):
        return cv2.Canny(self.img_obj, threshold1=self.threshold[0], threshold2=self.threshold[1],
                         apertureSize=self.aperture_size, L2gradient=self.L2Gradient)


# TRANSFORMATION

# resize
class ResizeTransform(ImageTransform):
    def __init__(self, src: str, size: tuple[int, int]):
        super().__init__(src, "pil")
        self.size = size

    def transform(self):
        return np.array(self.img_obj.resize(self.size))


# Cropping
class CropTransform(ImageTransform):
    def __init__(self, src: str, box: tuple[int, int, int, int]):
        super().__init__(src, "pil")
        self.box = box

    def transform(self):
        return np.array(self.img_obj.crop(self.box))


# Cropping Width
class CropWidthTransform(ImageTransform):
    def __init__(self, src: str, size: int):
        super().__init__(src, "pil")
        self.size = size

    def transform(self):
        width, height = self.img_obj.size
        return np.array(self.img_obj.crop((0, 0, width - self.size, height)))


# Center Cropping
class CenterCropTransform(ImageTransform):
    def __init__(self, src: str, size: tuple[int, int]):
        super().__init__(src, "pil")
        self.size = size

    def transform(self):
        transform = transforms.CenterCrop(self.size)
        return np.array(transform(self.img_obj))


# Random Cropping
class RandomCropTransform(ImageTransform):
    def __init__(self, src: str, size: tuple[int, int], padding: int):
        super().__init__(src, "pil")
        self.size = size
        self.padding = padding

    def transform(self):
        transform = transforms.RandomCrop(self.size, padding=self.padding)
        return np.array(transform(self.img_obj))


# Padding
class PaddingTransform(ImageTransform):
    def __init__(self, src: str, padding, size):
        super().__init__(src, "pil")
        self.padding = padding
        self.size = size

    def transform(self):
        if isinstance(self.size, int) and self.size == -1:
            if isinstance(self.padding, int):
                transform = transforms.Pad(self.padding)
                return np.array(transform(self.img_obj))
            elif isinstance(self.padding, tuple):
                width, height = self.img_obj.size
                # Left: self.padding[0], right: self.padding[1], bottom: self.padding[2], top: self.padding[3]
                new_width = width + self.padding[1] + self.padding[0]
                new_height = height + self.padding[3] + self.padding[2]
                result = Image.new(self.img_obj.mode, (new_width, new_height), (0, 0, 0))
                result.paste(self.img_obj, (self.padding[0], self.padding[1]))
                return np.array(result)
        elif isinstance(self.size, tuple):
            self.img_obj.thumbnail((self.size[0], self.size[1]))
            # print(img.size)
            delta_width = self.size[0] - self.img_obj.size[0]
            delta_height = self.size[1] - self.img_obj.size[1]
            pad_width = delta_width // 2
            pad_height = delta_height // 2
            padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
            return np.array(ImageOps.expand(self.img_obj, padding))
        return []


# Random Horizontal Flip
class RandomHorizontalFlipTransform(ImageTransform):
    def __init__(self, src: str, probability=0.5):
        super().__init__(src, "pil")
        self.probability = probability

    def transform(self):
        transform = transforms.RandomHorizontalFlip(self.probability)
        return np.array(transform(self.img_obj))


# Random Vertical Flip
class RandomVerticalFlipTransform(ImageTransform):
    def __init__(self, src: str, probability=0.5):
        super().__init__(src, "pil")
        self.probability = probability

    def transform(self):
        transform = transforms.RandomVerticalFlip(self.probability)
        return np.array(transform(self.img_obj))


# Random Perspective
class RandomPerspective(ImageTransform):
    def __init__(self, src: str, distortion_scale=0.5, probability=0.5):
        super().__init__(src, "pil")
        self.probability = probability
        self.distortion_scale = distortion_scale

    def transform(self):
        transform = transforms.RandomPerspective(distortion_scale=self.distortion_scale, p=self.probability)
        return np.array(transform(self.img_obj))
