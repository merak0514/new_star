# !/user/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def compute_diff(image1, image2):
    """计算差值,小于0归零"""
    image = np.array(image1, np.int) - np.array(image2, np.int)
    image *= (image > 0)
    image = np.array(image, np.uint8)
    return image


def cut_too_small(image, lambda_=1.5):
    """去掉过小的点（噪声）"""
    aver = np.sum(image)/np.sum(image != 0)
    image *= (image > aver * lambda_)
    return image


def cut_too_large(image):
    """超过一定阈值的点设为255: 避免出现星星变亮导致插值图值很大"""
    image_ = np.array((image < 150) * image, dtype=np.uint8)
    temp = np.array((image > 150) * 255, dtype=np.uint8)
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
            image[i + 1, j + 1] = np.sort(np.hstack(image[i:i + 3, j:j + 3]))[4]

    return image


def adjust_average(image1, image2):
    """
    adjust the average of image1 to fit image2
    note: only count non-zero
    :param image1:
    :type image1: np.array
    :param image2:
    :type image2: np.array
    :return: new image1
    :rtype: np.array
    """
    av1 = np.sum(image1)/np.sum(image1 != 0)
    av2 = np.sum(image2)/np.sum(image2 != 0)
    image1 = np.array(image1, np.float32)
    scale = av2 / av1
    print(scale)
    image1 *= scale
    image1 = np.array(np.rint(image1), np.uint8)
    return image1


if __name__ == '__main__':
    a = np.array([[1, 3], [2, 4]])
    b = np.array([[10, 20], [30, 40]])
    print(adjust_average(a, b))
