import matplotlib.pyplot as plt
from pathlib import PurePath
import numpy as np
import itertools
import skimage
import json
import math
import cv2
import os

import util

def add_border(im):
    row, col = im.shape[:2]
    bottom = im[row-2:row, 0:col]
    mean = cv2.mean(bottom)[0]

    border_size = 3
    border = cv2.copyMakeBorder(
        im,
        top=border_size,
        bottom=border_size,
        left=border_size,
        right=border_size,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )
    return border


class Preprocessing:
    def __init__(self, im_paths, output_filename, enable_visualization, debug_show_plots):
        self.debug_show_plots = debug_show_plots
        self.dir_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.im_paths = im_paths
        self.output_filename = output_filename
        self.filenames = [PurePath(path).parts[-1] for path in im_paths]
        self.imgs = [cv2.imread(im_path) for im_path in im_paths]

        self.contours_list = []
        self.meta = []

        for idx, img in enumerate(self.imgs):
            self.meta.append({
                'filename': self.filenames[idx],
                'top': [],
                'bottom': [],
                'inc_left': [],
                'inc_right': []
            })
            self.contours_list.append([])
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = add_border(img)
            
            # get thresholded image
            thresh = self.thresholding_plus(img_gray)

            # run watershedding algorithm (mostly for visualization right now, markers can be used for further analysis though)
            watershed, markers = self.test_watershed(thresh, img)

            # run contour algorithm, creating a visualization and saving the contours in a list
            contours = self.test_contours(thresh, idx)
            contours_markers = self.test_contours(markers, idx)
            
            # the following can be used to transform image with unconnected stitches to connected stitches
            # This allows us to use different kinds of analyses on the image, like hough transform and hit or miss
            kernel = np.ones((5,5),np.uint8)
            dil = cv2.dilate(thresh, kernel, iterations=6)
            erode = cv2.erode(dil, kernel, iterations=5)

            # The hough transformation isnt that stable, finding usable parameters would be necessary to make this a viable option
            # when stitches arent at 90 or zero degrees, it struggles to find the correct hough lines.
            skel_normal = self.test_skel_hough(thresh) # run hough transform with thresholded image
            hough_erode = self.test_skel_hough(erode) # run hough transform with dilated/eroded image

            # Basic hit or miss implementation, with a few more tweaks, this could be really useful
            self.test_hit_miss(erode, idx)
            
            if enable_visualization:
                self.visualize(thresh, self.filenames[idx], "thresholded")
                self.visualize(skel_normal, self.filenames[idx], "hough_transform_normal")
                self.visualize(hough_erode, self.filenames[idx], "hough_transform_eroded")
                self.visualize(watershed, self.filenames[idx], "watershedding")
                self.visualize(markers, self.filenames[idx], "watershedding_markers")
                self.visualize(contours, self.filenames[idx], "contours")
                self.visualize(contours_markers, self.filenames[idx], "contours_markers")
                self.visualize_hit_miss(thresh, idx, self.filenames[idx], "hit_or_miss")
        
        self.write_to_json()
        
        #self.find_incisions()


    # This function has to be improved a lot to also detect images with low contrast etc.
    def thresholding_plus(self, img):
        # Apply Gaussian blur for noise reduction
        blurred = cv2.GaussianBlur(img, (5, 5), 0)

        # Use adaptive thresholding to filter for incisions and stitches
        thresholding = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 18) # 35, 18

        # Reverse black/white values
        inverse = np.invert(thresholding)

        # add border to rule out some errors in futher analysis
        inverse_border = add_border(inverse)



        return inverse_border


    # actually pretty good, outliers must be removed from skeletonized image though.
    def test_hit_miss(self, img, idx):
        im_shape = img.shape
        height = im_shape[0]
        width = im_shape[1]

        # skeletonize preprocessed image
        skel = skimage.morphology.skeletonize(img)
        skel = skel.astype(np.uint8)

        # closing to get rid of unnecessary gaps
        kernel = np.ones((3, 3), np.uint8)
        skel_close = cv2.morphologyEx(skel, cv2.MORPH_CLOSE, kernel)
        skel_close = add_border(skel_close)
        im_shape = skel_close.shape
        height = im_shape[0]
        width = im_shape[1]

        # create structuring elements for top/bottom stitches and incision
        # These may have to be adjusted, so it finds the stitches more reliably
        top_kernel = np.zeros((3, 3), np.uint8)
        top_kernel[2][1] = 1

        bottom_kernel = np.zeros((3, 3), np.uint8)
        bottom_kernel[0][1] = 1

        incision_right_kernel = np.zeros((3, 3), np.uint8)
        incision_right_kernel[1][0] = 1
        #incision_right_kernel[1][1] = 1

        incision_left_kernel = np.zeros((3, 3), np.uint8)
        incision_left_kernel[1][2] = 1
        #incision_left_kernel[1][1] = 1
    
        # Use hit or miss strategy to find the ends of the stitches and incisions and save them in dictionary
        for y in range(1, height - 2):
            for x in range(1, width - 2):
                current_square = skel_close[y-1:y+2, x-1:x+2]
                if np.equal(current_square, top_kernel).all():
                    self.meta[idx]['top'].append([x,y])
                elif np.equal(current_square, bottom_kernel).all():
                    self.meta[idx]['bottom'].append([x,y])
                elif np.equal(current_square, incision_left_kernel).all():
                    self.meta[idx]['inc_left'].append([x,y])
                elif np.equal(current_square, incision_right_kernel).all():
                    self.meta[idx]['inc_right'].append([x,y])

        self.sort_and_filter_meta(idx)


    def sort_and_filter_meta(self, idx):
        # filter out unrealistic incision parts 
        self.meta[idx]['inc_left'].sort()
        self.meta[idx]['inc_right'].sort()

        self.meta[idx]['inc_left'] = self.meta[idx]['inc_left'][0]
        self.meta[idx]['inc_right'] = self.meta[idx]['inc_right'][-1]

        # sort stitches by x value
        self.meta[idx]['top'].sort()
        self.meta[idx]['bottom'].sort()

        # filter out top and bottom stitches that dont have a corresponding entry in each other's lists
        # some information gets lost by that, but necessary for further analysis
        top = self.meta[idx]['top']
        bottom = self.meta[idx]['bottom']
        if len(top) > len(bottom):
            idxs_to_remove = self.filter_stitches_old(bottom, top, np.abs(len(top) - len(bottom)))
            #for idx in idxs_to_remove:
            top = [v for i, v in enumerate(top) if i not in idxs_to_remove]
            self.meta[idx]['top'] = top
        elif len(top) < len(bottom):
            idxs_to_remove = self.filter_stitches_old(top, bottom, np.abs(len(top) - len(bottom)))
            #for idx in idxs_to_remove:
                #del bottom[idx]
            bottom = [v for i, v in enumerate(bottom) if i not in idxs_to_remove]
            self.meta[idx]['bottom'] = bottom

        self.sort_stitches(top, bottom, idx)


    def sort_stitches(self, top, bottom, idx):
        sorted_top = self.meta[idx]['top'].copy()
        sorted_bottom = []
        for i in range(len(top)):
            shortest_distance = float('inf')
            shortest_idx = 0
            for j in range(len(bottom)):
                dst = self.calculate_distance(top[i], bottom[j])
                if dst < shortest_distance:
                    shortest_distance = dst
                    shortest_idx = j
            if bottom[shortest_idx] not in sorted_bottom:
                #sorted_top.append(top[shortest_idx])
                sorted_bottom.append(bottom[shortest_idx])
            else:
                idx_to_replace = sorted_bottom.index(bottom[shortest_idx])
                if self.calculate_distance(self.meta[idx]['top'][idx_to_replace], sorted_bottom[idx_to_replace]) < shortest_distance:
                    sorted_bottom[idx_to_replace] = bottom[shortest_idx]
                    #sorted_top[idx_to_replace] = top[shortest_idx]
                del sorted_top[idx_to_replace]


        self.meta[idx]['bottom'] = sorted_bottom
        self.meta[idx]['top'] = sorted_top


    def calculate_distance(self, x1, x2):
        return abs(x1[0] - x2[0])


    def filter_stitches_old(self, shorter_list, longer_list, size_difference):
        distances = []
        shorter_list = np.asarray(shorter_list)
        for stitch in longer_list:
            distances.append(min(np.abs(shorter_list[:,0] - stitch[0])))

        distances = np.asarray(distances)
        idxs_to_remove = np.argpartition(distances, -size_difference)[-size_difference:]

        return idxs_to_remove

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
    def test_contours(self, inverse, idx):
        contours, _ = cv2.findContours(inverse, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 0 and area > 10:
                filtered_contours.append(contour)

        self.contours_list[idx].append(filtered_contours)
        inverse_new = cv2.merge([inverse,inverse, inverse])
        for contour in filtered_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(inverse_new, (x, y), (x + w, y + h), (255, 0, 0), 2)



        return inverse_new



    def write_to_json(self):
        output_json = []
        for idx, im_meta in enumerate(self.meta):
            #crossing_angles = calc_angles()
            output_json.append({
                "filename": im_meta['filename'],
                "incision_polyline": [im_meta['inc_left'], im_meta['inc_right']],
                "crossing_positions": [],
                "crossing_angles": []
            })
            for top, bottom in zip(im_meta['top'], im_meta['bottom']):
                # calculate crossing position as pixels from the start of the incision (lefthand side)
                intersect = util.intersectLines(top, bottom, im_meta['inc_left'], im_meta['inc_right'])
                crossing_position = intersect[0] - im_meta['inc_left'][0]
                output_json[idx]['crossing_positions'].append(crossing_position)
                # calculate angle between stitch and incision
                crossing_angle = util.ang([top, bottom], [im_meta['inc_left'], im_meta['inc_right']])
                output_json[idx]['crossing_angles'].append(crossing_angle)

        with open(os.path.join(self.dir_path, "out", self.output_filename), 'w', encoding='utf-8') as f:
            json.dump(output_json, f, ensure_ascii=False, indent=4)
        test = 0


    def visualize(self, img, im_name, method_name):
        #x = [self.incision[0][0][0], self.incision[0][1][0]]
        #y = [self.incision[0][0][1], self.incision[0][1][1]]
        #plt.plot(x, y, 'r')
        plt.imshow(img)
        plt.title(method_name + " " + im_name)
        plt.savefig(os.path.join(self.dir_path, "out", (method_name + "_" + im_name)))
        if self.debug_show_plots:
            plt.show()
        plt.close()


    def visualize_hit_miss(self, img, idx, im_name, method_name):
        # draw incision
        img_color = cv2.merge([img, img, img])
        cv2.line(img_color, self.meta[idx]['inc_left'], self.meta[idx]['inc_right'], (255,0,0), 2)

        # draw stitches
        for i in range(len(self.meta[idx]['top'])):
            cv2.line(img_color, self.meta[idx]['top'][i], self.meta[idx]['bottom'][i], (255,0,0), 2)
        plt.imshow(img_color)
        plt.title(method_name + " " + im_name)
        plt.savefig(os.path.join(self.dir_path, "out", (method_name + "_" + im_name)))
        if self.debug_show_plots:
            plt.show()
        plt.close()
