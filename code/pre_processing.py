# !/user/bin/env python
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import csv
train_image_path = '../af2019-cv-training-20190312/'
train_label_path = '../af2019-cv-training-20190312/list.csv'
a = '_a'
b = '_b'
c = '_c'
end = '.jpg'


def cut_too_small(image, lambda_=1.5):
    """去掉过小的点（噪声）"""
    aver = np.average(image)
    image *= (image > aver*lambda_)
    return image


def cut_too_large(image):
    """超过一定阈值的点设为255: 避免出现星星变亮导致插值图值很大"""
    image_ = np.array((image < 150) * image, dtype=np.float32)
    temp = np.array((image > 150) * 255, dtype=np.float32)
    image_ += temp
    return image_


def middle_filter(image):
    """中值滤波: 把九个点的中位数作为中间点的值"""
    x_len = np.size(image, 0)
    y_len = np.size(image, 1)
    for i in range(x_len - 2):
        for j in range(y_len - 2):
            # if np.sum(image[i:i+3, j:j+3]) == image[i+1, j+1]:
            #     image[i+1, j+1] = 0
            image[i + 1, j + 1] = np.sort(np.hstack(image[i:i+3, j:j+3]))[4]

    return image