import matplotlib.pyplot as plt
import skimage
import cv2
import numpy as np
import math


class Preprocessing:
    def __init__(self, im_paths):
        self.im_paths = im_paths
        self.imgs = [cv2.imread(im_path) for im_path in im_paths]
        self.incision = []
        self.stitches = []
        for img in self.imgs:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh = self.thresholding_plus(img_gray)

            #skel = self.test_skel_hough(thresh)
            watershed, markers = self.test_watershed(thresh, img)
            contours = self.test_contours(thresh)
            
            # the following can be used to transform image with unconnected stitches to connected stitches
            # The hough transformation isnt that stable though, hard to find out why (probably has to do with the combination of parameters)
            # when stitches arent perpendicular, 
            kernel = np.ones((5,5),np.uint8)
            dil = cv2.dilate(markers, kernel, iterations=6)
            erode = cv2.erode(dil, kernel, iterations=6)
            hough_erode = self.test_skel_hough(erode)

            self.visualize(hough_erode)
            self.visualize(thresh)
            self.visualize(skel)
            self.visualize(watershed)
            self.visualize(contours)
            self.visualize(markers)
            self.visualize(self.test_contours(markers))



    def test_sobel_hor_ver(self, img):
        edge_hor = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        edge_hor_thresh = edge_hor > 0
        edge_ver = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        edge_ver_thresh = edge_ver > 0


    # If the image shows a fully stitched incision, using other grouping methods isnt viable
    # Extracting horizontal and vertical edges, the incision can be seperated from the stitches, and grouped by analyzing the angle
    def test_skel_hough(self, img):

        # skeletonize preprocessed image
        skel = skimage.morphology.skeletonize(img)
        skel = skel.astype(np.uint8)

        # closing to get rid of unnecessary gaps
        kernel = np.ones((3, 3), np.uint8)
        skel_close = cv2.morphologyEx(skel, cv2.MORPH_CLOSE, kernel)
        plt.imshow(skel_close)

        # use hough transform to get all lines
        minLineLength=20
        maxLineGap = 30
        lines = cv2.HoughLinesP(image=skel_close, rho=1, theta=np.pi/20, threshold=10, lines=np.array([]), minLineLength=minLineLength, maxLineGap=maxLineGap)

        # write detected lines to image
        a,b,c = lines.shape
        img_new = cv2.merge([img, img, img])
        skel_new = cv2.merge([255*skel,255*skel,255*skel])
        for i in range(a):
            x1, y1, x2, y2 = lines[i][0]
            if x1 == x2:
                slope = 90
            else:
                slope = (y2 - y1) / (x2 - x1)
            angle = np.math.atan(slope) * 180. / np.pi
            if -10 < angle < 10:
                cv2.line(skel_new, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 1, cv2.LINE_AA)
            if 70 < abs(angle) < 110:
                cv2.line(skel_new, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (255, 0, 0), 1, cv2.LINE_AA)

        return skel_new


    # This seems to seperate the areas quite well, not sure if it's useful yet though. Might be worth looking into more
    def test_watershed(self, inverse, img_color):
        # noise removal
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(inverse,cv2.MORPH_OPEN,kernel, iterations = 2)

        # sure background area
        sure_bg = cv2.dilate(opening,kernel,iterations=4)

        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
        ret, sure_fg = cv2.threshold(dist_transform,0.2*dist_transform.max(),255,0)

        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg,sure_fg)


        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1
        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0
        markers_thresh = 255*(markers > 1).astype(np.uint8)

        markers_watershed = cv2.watershed(img_color, markers)
        img_color[markers_watershed == -1] = [255,0,0]

        return img_color, markers_thresh



    # Locates the incisions and stitches somewhat well, couple of errors though, not sure if they're fixed easily
    # Maybe combine with watershedding?
    # e.g watershedding to create markers, and use those as reference for the contours, would get rid of unnecessary noise
    # Maybe there's an easier way to 
    def test_contours(self, inverse):
        contours, _ = cv2.findContours(inverse, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 0 and area > 10:
                filtered_contours.append(contour)

        inverse_new = cv2.merge([inverse,inverse, inverse])
        for contour in filtered_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(inverse_new, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return inverse_new


    # This function has to be improved a lot to also detect images with low contrast etc.
    def thresholding_plus(self, img):
        # Apply Gaussian blur for noise reduction
        blurred = cv2.GaussianBlur(img, (5, 5), 0)

        # Use adaptive thresholding to filter for incisions and stitches
        thresholding = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 18) # 35, 18

        # Reverse black/white values
        inverse = np.invert(thresholding)

        return inverse



    def visualize(self, img):
        #x = [self.incision[0][0][0], self.incision[0][1][0]]
        #y = [self.incision[0][0][1], self.incision[0][1][1]]
        #plt.plot(x, y, 'r')
        plt.imshow(img)
        plt.show()


    '''
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
    '''