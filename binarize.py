
# coding: utf-8

# In[ ]:


import glob
import time
import os
import pickle
import cv2
import numpy as np


# In[ ]:


class binarize:
    def __init__(self, image):
        self.image=image
        
    def noise_reduction(self, image, threshold=4):

        """
        This method is used to reduce the noise of binary images.

        :param image:
            binary image (0 or 1)

        :param threshold:
            min number of neighbours with value

        :return:
        """
        k = np.array([[1, 1, 1],
                      [1, 0, 1],
                      [1, 1, 1]])
        image_copy1 = np.copy(image)
        nb_neighbours = cv2.filter2D(image_copy1, ddepth=-1, kernel=k)
        image_copy1[nb_neighbours < threshold] = 0
        return image_copy1
    
    def binarize(self, gray_thresh=(20, 255), s_thresh=(170, 255), l_thresh=(30, 255), sobel_kernel=3):
        """
        This method extracts lane line pixels from road an image. Then create a minarized image where
        lane lines are marked in white color and rest of the image is marked in black color.
        :param image:
            Source image
        :param gray_thresh:
            Minimum and maximum gray color threshold
        :param s_thresh:
            This tuple contains the minimum and maximum S color threshold in HLS color scheme
        :param l_thresh:
            Minimum and maximum L color (after converting image to HLS color scheme)
            threshold allowed in the source image
        :param sobel_kernel:
            Size of the kernel use by the Sobel operation.
        :return:
            The binarized image where lane line pixels are marked in while color and rest of the image
            is marked in block color.
        """
        # first we take a copy of the source iamge
        image_copy = np.copy(self.image)
        # convert RGB image to HLS color space.
        # HLS more reliable when it comes to find out lane lines
        hls = cv2.cvtColor(image_copy, cv2.COLOR_RGB2HLS)
        s_channel = hls[:, :, 2]
        l_channel = hls[:, :, 1]
        # Next, we apply Sobel operator in X direction and calculate scaled derivatives.
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        # Next, we generate a binary image based on gray_thresh values.
        thresh_min = gray_thresh[0]
        thresh_max = gray_thresh[1]
        sobel_x_binary = np.zeros_like(scaled_sobel)
        sobel_x_binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
        # Next, we generated a binary image using S component of our HLS color scheme and
        # provided S threshold
        s_binary = np.zeros_like(s_channel)
        s_thresh_min = s_thresh[0]
        s_thresh_max = s_thresh[1]
        s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
        # Next, we generated a binary image using S component of our HLS color scheme and
        # provided S threshold
        l_binary = np.zeros_like(l_channel)
        l_thresh_min = l_thresh[0]
        l_thresh_max = l_thresh[1]
        l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1
        # finally, return the combined binary image
        binary = np.zeros_like(sobel_x_binary)
        binary[((l_binary == 1) & (s_binary == 1) | (sobel_x_binary == 1))] = 1
        binary = 255 * np.dstack((binary, binary, binary)).astype('uint8')
        return self.noise_reduction(image=binary, threshold=4)

