# !/user/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 16:33
# @Author   : Merak
# @File     : test.py
# @Software : PyCharm
import cv2
import csv
import numpy as np
from pre_processing import *
import matplotlib.pyplot as plt
train_image_path = '../af2019-cv-training-20190312/'
train_label_path = '../af2019-cv-training-20190312/list.csv'
a = '_a'
b = '_b'
c = '_c'
end = '.jpg'


def get_pos_based_on_name(name):
    train_data = []
    with open(train_label_path) as csvfile:
        csv_reader = csv.reader(csvfile)
        data_header = next(csv_reader)
        for row in csv_reader:
            train_data.append(row)
    for datum in train_data:
        if datum[0] == name:
            return int(datum[1]), int(datum[2]), datum[3]


def observe_image_using_your_eyes():
    """临时之举"""
    img_name = 'cfd7249b14b8015977e4b2f2dd4eb775'

    imb = cv2.imread(''.join((train_image_path, img_name[:2], '/', img_name, b, end)))
    imb = np.array(imb[:, :, 0], np.uint8)
    imc = cv2.imread(''.join((train_image_path, img_name[:2], '/', img_name, c, end)))
    imc = np.array(imc[:, :, 0], np.uint8)

    imb = adjust_average(imb, imc)

    imb = cut_too_small(imb, lambda_=1.25)
    imb = cut_too_large(imb)
    # cv2.imshow('img_b', img_b)
    imb = middle_filter(imb)
    # cv2.imshow('imb2', img_b)

    imc = cut_too_small(imc, lambda_=1.25)
    imc = cut_too_large(imc)
    imc = middle_filter(imc)

    ima_cut = compute_diff(imb, imc)

    # cv2.imshow('ima_cut', ima_cut)
    x, y, label = get_pos_based_on_name(img_name)
    print('label', label)
    img_a = cv2.circle(ima_cut, (x, y), 15, 255, 1)

    img_a = cv2.rectangle(img_a, (x-30, y-30), (x+30, y+30), 255, 1)
    print(np.shape(imb))
    img_b = cv2.circle(imb, (x, y), 15, 255, 1)
    img_c = cv2.circle(imc, (x, y), 15, 255, 1)
    cv2.imshow('img_a', img_a)
    cv2.imshow('img_b', img_b)
    cv2.imshow('img_c', img_c)
    cv2.waitKey(0)


if __name__ == '__main__':
    image_name = '000db175d712996f1cfd20cc7d600223'
    img_b = cv2.imread(''.join((train_image_path, image_name[:2], '/', image_name, b, end)))
    img_b = np.array(img_b[:, :, 0], np.uint8)
    img_c = cv2.imread(''.join((train_image_path, image_name[:2], '/', image_name, c, end)))
    img_c = np.array(img_c[:, :, 0], np.uint8)
    img_b = adjust_average(img_b, img_c)
    img_b = cut_too_small(img_b, lambda_=1.25)
    img_b = cut_too_large(img_b)
    # cv2.imshow('img_b', img_b)
    img_b = middle_filter(img_b)
    # cv2.imshow('imb2', img_b)

    img_c = cut_too_small(img_c, lambda_=1.25)
    img_c = cut_too_large(img_c)
    img_c = middle_filter(img_c)

    print(np.shape(img_b))
    cut_images_b, cut_images_c = random_cut(img_b, img_c, (100, 100), 40, 10)
    # print(np.shape(cut_images_b))
    # cv2.imshow('b0', cut_images_b[6])
    # cv2.waitKey(0)
    path = ''.join(['../cut_data/', image_name[:2]])
    if not os.path.exists(path):
        print(path)
        os.makedirs(path)
    for i in range(len(cut_images_b)):
        image = cut_images_b[i]
        im_path = ''.join([path, '/', image_name, str(i), b, end])
        cv2.imwrite(im_path, image)
    for i in range(len(cut_images_c)):
        image = cut_images_c[i]
        im_path = ''.join([path, '/', image_name, str(i), c, end])
        cv2.imwrite(im_path, image)

    # observe_image_using_your_eyes()


