import matplotlib.pyplot as plt
import skimage
import cv2
import numpy as np


class Preprocessing:
    def __init__(self, im_paths):
        self.im_paths = im_paths
        self.imgs = [cv2.imread(im_path, 0) for im_path in im_paths]
        self.incision = []
        self.stitches = []
        for img in self.imgs:
            img_proc = self.thresholding_plus(img)
            self.incision.append(self.find_incision(img_proc))
            #self.stitches.append()
            self.visualize(img_proc)

    def thresholding_plus(self, img):
        # Apply Gaussian blur for noise reduction
        blurred = cv2.GaussianBlur(img, (5, 5), 0)

        # Use adaptive thresholding to filter for incisions and stitches
        thresholding = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 18)

        # Reverse black/white values
        inverse = np.invert(thresholding)       

        return inverse

    # function for extracting horizontal lines from image - not useful yet
    def horizontal_edges(self, img):
        edge_hor = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        _, edge_hor = cv2.threshold(edge_hor, 50, 255, cv2.THRESH_BINARY)
        plt.imshow(edge_hor, cmap="gray")

        return edge_hor

    def find_incision(self, img):
        height, width = img.shape[0], img.shape[1]
        leftmost_black = [None,None]
        rightmost_black = [None,None]
        for x in range(width - 1):
            for y in range(height - 1):
                # Check if the pixel is black (0)
                if img[y, x] == 255:
                    # Update leftmost and rightmost positions
                    if leftmost_black[0] is None or x < leftmost_black[0]:
                        leftmost_black = [x, y]
                    if rightmost_black[0] is None or x > rightmost_black[0]:
                        rightmost_black = [x, y]
        return [leftmost_black, rightmost_black]

    def find_stitches(self, img):
        stitches = [] 
        return stitches

    def find_stitches(self, img):
        return 0



    def visualize(self, img):
        x = [self.incision[0][0][0], self.incision[0][1][0]]
        y = [self.incision[0][0][1], self.incision[0][1][1]]
        plt.plot(x, y, 'r')
        plt.imshow(img, cmap="gray")
        plt.show()