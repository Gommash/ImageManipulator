import getopt
import os
import sys
import time
import logging

from PIL import Image

import filetypeconverter
import imagemanipulator
from imagemanipulator import RollingBallFilter


def main():
    helpinfo = 'imagetool.py -i <src_directory> -o <output_directory> -c <command>'
    idir, odir, command = "", "", ""

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hi:o:c:")
    except getopt.GetoptError:
        print(helpinfo)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(helpinfo)
            sys.exit()
        elif opt in ("-i", "--input"):
            idir = arg
            if not os.path.exists(idir):
                os.makedirs(idir)
        elif opt in ("-o", "--output"):
            odir = arg
            if not os.path.exists(odir):
                os.makedirs(odir)
        elif opt in ("-c", "--command"):
            command = arg.lower().split()

    print(f"idir={idir}, odir={odir}, command={command}")

    if command[0] == "filetype-converter":
        if command[1] == "hdf":
            filetypeconverter.HDFConverter(idir, odir, command[2]).convert()
        elif command[1] == "dicom":
            filetypeconverter.DicomConverter(idir, odir, command[2]).convert()
    else:
        for filename in os.listdir(idir):
            img_src = os.path.join(idir, filename)
            if img_src is not None:
                if os.path.isfile(img_src):
                    try:
                        img_data = 0
                        start = time.time()
                        # Rolling ball
                        if command[0] == "rolling-ball":
                            img_data = RollingBallFilter(img_src, radius=float(command[1]), invert=bool(command[2] == "true"),
                                                         filepath=f"{odir}\\{filename}").enhance()
                        # Color
                        elif command[0] == "color":
                            img_data = imagemanipulator.ColorFilter(img_src, float(command[1])).enhance()
                        # Sharpness
                        elif command[0] == "sharpness":
                            img_data = imagemanipulator.SharpnessFilter(img_src, float(command[1])).enhance()
                        # Brightness
                        elif command[0] == "brightness":
                            img_data = imagemanipulator.BrightnessFilter(img_src, float(command[1])).enhance()
                        # Contrast
                        elif command[0] == "contrast":
                            img_data = imagemanipulator.ContrastFilter(img_src, float(command[1])).enhance()
                        # Gaussian Blur
                        elif command[0] == "gaussian-blur":
                            img_data = imagemanipulator.GaussianBlurFilter(img_src, blurgrid=eval(command[1])).enhance()
                        # Clahe
                        elif command[0] == "clahe":
                            img_data = imagemanipulator.CLAHEFilter(img_src, cliplimit=eval(command[1]),
                                                                    tilegridsize=eval(command[2]),
                                                                    factor=eval(command[3])).enhance()
                        # Histogram Equalization
                        elif command[0] == "histogram-equalization":
                            img_data = imagemanipulator.HistogramEqualizationFilter(img_src).enhance()
                        # NL Means Denoising
                        elif command[0] == "nl-means-denoise":
                            img_data = imagemanipulator.DenoiseNLMeansFilter(img_src,
                                                                             patch_size=int(command[1]),
                                                                             patch_distance=int(command[2]),
                                                                             h=float(command[3]),
                                                                             fast_mode=bool(command[4]) == "true").enhance()
                        # Wiener Denoising
                        elif command[0] == "wiener":
                            img_data = imagemanipulator.DenoiseWienerFilter(img_src, balance=int(command[1])).enhance()
                        # Richardson Lucy Deblur / Denoise
                        elif command[0] == "richardson-lucy":
                            img_data = imagemanipulator.RichardsonLucyFilter(img_src, num_iter=int(command[1])).enhance()
                        # Wavelet Denoising
                        elif command[0] == "wavelet":
                            img_data = imagemanipulator.DenoiseWaveletFilter(img_src, sigma=float(command[1]),
                                                                             rescale_sigma=bool(command[2]) == "true").enhance()
                        # Bilateral Denoising
                        elif command[0] == "bilateral":
                            img_data = imagemanipulator.DenoiseBilateralFilter(img_src, sigma_color=float(command[1]),
                                                                               sigma_spatial=bool(command[2]) == "true").enhance()
                        # TvBergman Denoising
                        elif command[0] == "tv-bergman":
                            img_data = imagemanipulator.DenoiseTvBregmanFilter(img_src, weight=float(command[1]),
                                                                               max_num_iter=int(command[2]),
                                                                               isotropic=bool(command[3]) == "true").enhance()
                        # TvBergman Denoising
                        elif command[0] == "tv-chambolle":
                            img_data = imagemanipulator.DenoiseTvChambolleFilter(img_src, weight=float(command[1]),
                                                                                 max_num_iter=int(command[2]),
                                                                                 isotropic=bool(command[3]) == "true").enhance()
                        # Canny Edge Detection
                        elif command[0] == "canny":
                            img_data = imagemanipulator.CannyEdgeDetectionFilter(img_src, threshold=eval(command[1]),
                                                                                 aperture_size=int(command[2]),
                                                                                 l2gradient=bool(command[3]) == "true").enhance()
                        # Laplacian Edge Detection
                        elif command[0] == "laplacian":
                            img_data = imagemanipulator.LaplacianEdgeDetectionFilter(img_src, ksize=int(command[1]),
                                                                                     delta=int(command[2]),
                                                                                     scale=int(command[3])).enhance()
                        # Sobel X Edge Detection
                        elif command[0] == "sobel":
                            img_data = imagemanipulator.SobelEdgeDetectionFilter(img_src, xaxis=bool(command[1]) == "true").enhance()
                        # Random Perspective
                        elif command[0] == "random-perspective":
                            img_data = imagemanipulator.RandomPerspective(img_src, distortion_scale=float(command[1]),
                                                                          probability=float(command[2])).transform()
                        # Random Horizontal Flipping
                        elif command[0] == "random-horizontal-flip":
                            img_data = imagemanipulator.RandomHorizontalFlipTransform(img_src,
                                                                                      probability=float(
                                                                                          command[1])).transform()
                        # Random Vertical Flipping
                        elif command[0] == "random-vertical-flip":
                            img_data = imagemanipulator.RandomVerticalFlipTransform(img_src,
                                                                                    probability=float(
                                                                                        command[1])).transform()
                        # Cropping width
                        elif command[0] == "crop-width":
                            img_data = imagemanipulator.CropWidthTransform(img_src, size=int(command[1])).transform()
                        # Cropping
                        elif command[0] == "crop":
                            img_data = imagemanipulator.CropTransform(img_src, box=eval(command[1])).transform()
                        # Random Cropping
                        elif command[0] == "random-crop":
                            img_data = imagemanipulator.RandomCropTransform(img_src, size=eval(command[1]),
                                                                            padding=int(command[2])).transform()
                        # Center Cropping
                        elif command[0] == "center-crop":
                            img_data = imagemanipulator.CenterCropTransform(img_src, size=eval(command[1])).transform()
                        # Resize
                        elif command[0] == "resize":
                            img_data = imagemanipulator.ResizeTransform(img_src, size=eval(command[1])).transform()
                        # padding
                        elif command[0] == "padding":
                            img_data = imagemanipulator.PaddingTransform(img_src, padding=eval(command[1]),
                                                                         size=eval(command[2])).transform()
                        if not isinstance(img_data, int):
                            img = Image.fromarray(img_data)
                            img.convert('RGB').save(f"{odir}\\{filename}")
                        print(
                            f"{command[0]} was applied on: {odir}\\{filename} time taken: {(time.time() - start) * 1000}ms",
                            flush=True)
                    except Exception as inst:
                        print(type(inst))  # the exception instance
                        print(inst)
                        print(f"Failed to apply filter on: {img_src}")
                        # Create and configure logger
                        logging.basicConfig(filename="err.log",
                                            format='%(asctime)s %(message)s',
                                            filemode='w')
                        # Creating an object
                        logger = logging.getLogger()
                        # Setting the threshold of logger to DEBUG
                        logger.setLevel(logging.ERROR)
                        # Test messages
                        logger.error(f"type: {type(inst)}, inst: {inst}, Message: Failed to apply filter on: {img_src}")


if __name__ == '__main__':
    main()
