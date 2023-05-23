import matplotlib.pyplot as plt
import skimage
import cv2
import numpy as np
import math


class Preprocessing:
    def __init__(self, im_paths, output_filename):
        self.im_paths = im_paths
        self.filenames = [path.split("/")[-1] for path in im_paths]
        self.output_filename = output_filename
        self.imgs = [cv2.imread(im_path) for im_path in im_paths]
        self.incisions = []
        self.stitches = []
        self.contours_list = []
        for img in self.imgs:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh = self.thresholding_plus(img_gray)

            skel = self.test_skel_hough(thresh)
            watershed, markers = self.test_watershed(thresh, img)
            contours, areas = self.test_contours(thresh)
            self.contours_list.append(areas)
            
            # the following can be used to transform image with unconnected stitches to connected stitches
            # The hough transformation isnt that stable though, hard to find out why (probably has to do with the combination of parameters)
            # when stitches arent perpendicular, it struggles to find the correct hough lines.
            kernel = np.ones((5,5),np.uint8)
            dil = cv2.dilate(markers, kernel, iterations=6)
            erode = cv2.erode(dil, kernel, iterations=6)
            hough_erode = self.test_skel_hough(erode)
            self.hit_miss = self.test_hit_miss(erode)
            '''
            self.visualize(hough_erode)
            self.visualize(thresh)
            self.visualize(skel)
            self.visualize(watershed)
            self.visualize(contours)
            self.visualize(markers)
            contours_markers, areas_markers = self.test_contours(markers)
            self.visualize(contours_markers)
            '''
            self.draw_hit_miss(self.hit_miss, thresh)
        
        #self.find_incisions()


    def draw_hit_miss(self, meta, img):
        # draw incision
        img_color = cv2.merge([img, img, img])
        cv2.line(img_color, meta['inc_left'][0], meta['inc_right'][0], (255,0,0), 2)

        # draw stitches
        for i in range(len(meta['top'])):
            cv2.line(img_color, meta['top'][i], meta['bottom'][i], (255,0,0), 2)
        plt.imshow(img_color)
        plt.show()
        return

    def find_incisions(self):
        self.incisions = []
        for contours in self.contours_list:
            areas = [cv2.contourArea(contour) for contour in contours]
            incision_idx = areas.index(max(areas))
            self.incisions.append(contours[incision_idx])
        return

    def write_to_json(self):
        output_json = {}
        for filename in self.filenames:
            return

    # actually pretty good, outliers must be removed from skeletonized image though.
    def test_hit_miss(self, img):
        im_shape = img.shape
        height = im_shape[0]
        width = im_shape[1]

        # skeletonize preprocessed image
        skel = skimage.morphology.skeletonize(img)
        skel = skel.astype(np.uint8)

        # closing to get rid of unnecessary gaps
        kernel = np.ones((3, 3), np.uint8)
        skel_close = cv2.morphologyEx(skel, cv2.MORPH_CLOSE, kernel)

        # create structuring elements for top/bottom stitches and incision
        top_kernel = np.zeros((3, 3), np.uint8)
        top_kernel[2][1] = 1

        bottom_kernel = np.zeros((3, 3), np.uint8)
        bottom_kernel[0][1] = 1

        incision_left_kernel = np.zeros((3, 3), np.uint8)
        incision_left_kernel[1][0] = 1
        incision_left_kernel[1][1] = 1

        incision_right_kernel = np.zeros((3, 3), np.uint8)
        incision_right_kernel[1][2] = 1
        incision_right_kernel[1][1] = 1
        


        # Use hit or miss strategy to find the ends of the stitches and incisions and save them in dictionary
        meta = {
            'top': [],
            'bottom': [],
            'inc_left': [],
            'inc_right': []
        }
        for y in range(1, height - 2):
            for x in range(1, width - 2):
                current_square = skel_close[y-1:y+2, x-1:x+2]
                if np.equal(current_square, top_kernel).all():
                    meta['top'].append([x,y])
                elif np.equal(current_square, bottom_kernel).all():
                    meta['bottom'].append([x,y])
                elif np.equal(current_square, incision_left_kernel).all():
                    meta['inc_left'].append([x,y])
                elif np.equal(current_square, incision_right_kernel).all():
                    meta['inc_right'].append([x,y])

        # maybe add function to filter out unrealistic incision parts here

        # sort stitches by x value
        meta['top'].sort()
        meta['bottom'].sort()

        plt.imshow(skel_close)
        plt.show()

        return meta

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
    def test_watershed(self, inverse, img):
        img_color = img.copy()
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

        return inverse_new, filtered_contours


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