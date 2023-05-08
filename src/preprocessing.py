import matplotlib.pyplot as plt
import skimage
import cv2
import numpy as np


class Preprocessing:
    def __init__(self, im_paths):
        self.im_paths = im_paths
        self.imgs = [skimage.io.imread(im_path, as_gray=True) for im_path in im_paths]
        for img in self.imgs:
            img_proc = self.edge_detection(img)
            self.visualize(img_proc)

    def edge_detection(self, img):
        canny = skimage.feature.canny(img)
        canny = self.morph_closing(canny)
        return canny

    def morph_closing(self, img):
        morphc = skimage.morphology.binary_closing(img, np.ones((5,5)))
        morphc = skimage.morphology.area_closing(morphc)

        return morphc

    def visualize(self, img):
        plt.imshow(img, cmap="gray")