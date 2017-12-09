
# coding: utf-8

# In[3]:


import glob
import time
import os
import pickle
import cv2
import numpy as np


# In[4]:


class PerspectiveTransformer:
    def __init__(self, src_points, dest_points):
        """
        :param src_points:
        :param dest_points:
        """
        self.src_points = src_points
        self.dest_points = dest_points
        self.M = cv2.getPerspectiveTransform(self.src_points, self.dest_points)
        self.M_inverse = cv2.getPerspectiveTransform(self.dest_points, self.src_points)
        
    def transform(self, image):
        """
        :param image:
        :return:
        """
        size = (image.shape[1], image.shape[0])
        return cv2.warpPerspective(image, self.M, size, flags=cv2.INTER_LINEAR)

    def inverse_transform(self, src_image):
        """
        :param src_image:
        :return:
        """
        size = (src_image.shape[1], src_image.shape[0])
        return cv2.warpPerspective(src_image, self.M_inverse, size, flags=cv2.INTER_LINEAR)

