import os

import cv2
import h5py
import numpy as np
import pandas as pd
import pydicom as dicom
from PIL import Image
from tqdm import tqdm


class ImageFileTypeConverter:
    def __init__(self, inputdirectory: str, outputdirectory: str, img_filetype: str):
        self.inputdirectory = inputdirectory
        self.outputdirectory = outputdirectory
        self.img_filetype = img_filetype

    def convert(self):
        pass


class HDFConverter(ImageFileTypeConverter):
    def __init__(self, inputdirectory: str, outputdirectory: str, img_filetype: str):
        super().__init__(inputdirectory, outputdirectory, img_filetype)

    def convert(self):
        for filename in os.listdir(self.inputdirectory):
            img_src = os.path.join(self.inputdirectory, filename)
            if img_src is not None:
                if os.path.isfile(img_src) and os.path.splitext(img_src)[1] == ".h5":
                    img_name = os.path.splitext(filename)[0].split('_')
                    if not os.path.exists(self.outputdirectory + f"\\{img_name[-2]}\\{img_name[-1]}"):
                        os.makedirs(self.outputdirectory + f"\\{img_name[-2]}\\{img_name[-1]}")

                    print("Extracting images")
                    hdf_file = h5py.File(img_src, "r")
                    data = hdf_file[img_name[-1]]
                    for idx, img in tqdm(enumerate(data), total=len(hdf_file[img_name[-1]])):
                        if img_name[-1] == "x" or img_name[-1] == "mask":
                            cv2.imwrite(self.outputdirectory + "\\" + img_name[
                                -2] + f"\\{img_name[-1]}\\{idx}." + self.img_filetype,
                                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                        elif img_name[-1] == "y":
                            hdf_file = h5py.File(img_src, "r")
                            data = hdf_file[img_name[-1]]
                            out = np.zeros(data.shape, dtype=np.uint8)
                            data.read_direct(out)
                            pd.DataFrame(out.ravel()).to_csv(
                                self.outputdirectory + f"\\{img_name[-2]}\\{img_name[-1]}\\{idx}.csv", header=None,
                                index=None)


class DicomConverter(ImageFileTypeConverter):
    def __init__(self, inputdirectory: str, outputdirectory: str, img_filetype: str):
        super().__init__(inputdirectory, outputdirectory, img_filetype)

    def read_files(self, directory):
        for filename in os.listdir(directory):
            src = os.path.join(directory, filename)
            if src is not None:
                if os.path.isfile(src) and os.path.splitext(src)[1] == ".dcm":
                    self.convert_image_file(src)
                elif os.path.isdir(src):
                    self.read_files(src)

    def convert_image_file(self, image_file):
        ds = dicom.dcmread(image_file)
        # Convert to float to avoid overflow or underflow losses.
        image = ds.pixel_array.astype(float)
        img = (np.maximum(image, 0)/image.max())*255
        final = np.uint8(img)
        final = Image.fromarray(final)
        imagepath = image_file.replace('.dcm', f".{self.img_filetype}")
        final.save(imagepath)

    def convert(self):
        self.read_files(self.inputdirectory)

        for filename in os.listdir(self.inputdirectory):
            src = os.path.join(self.inputdirectory, filename)
            if src is not None:
                if os.path.isfile(src) and os.path.splitext(src)[1] == ".dcm":
                    if not os.path.exists(self.outputdirectory):
                        os.makedirs(self.outputdirectory)

                # Specify the .dcm folder path
                images_path = os.listdir(self.inputdirectory)
                for n, image in enumerate(images_path):
                    ds = dicom.dcmread(os.path.join(self.inputdirectory, image))
                    # Convert to float to avoid overflow or underflow losses.
                    image_2d = ds.pixel_array.astype(float)
                    # Rescaling grey scale between 0-255
                    image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 256
                    # Convert to uint
                    image_2d_scaled = np.uint8(image_2d_scaled)
                    image = image.replace('.dcm', f".{self.img_filetype}")
                    cv2.imwrite(os.path.join(self.outputdirectory, image), image_2d_scaled)
