# !/user/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 15:01
# @Author   : Merak
# @File     : run.py
# @Software : PyCharm
import classification  # 要改
import torch
import csv
import pre_processing
import os
import numpy as np
import cv2
b = '_b'
c = '_c'
end = '.jpg'

data_set_list_path = '../../testb/list.csv'
data_set_path = '../../testb/'
model_path = '../classification/model2/'
data_set = []
with open(data_set_list_path) as f:
    csv_reader = csv.reader(f)
    for row in csv_reader:
        data_set.append(row[0])

print('The length of train_data is {}'.format(len(data_set)))

if input('Print y to continue') is not 'y':
    exit()

csv_file = open('../cut_data/labels.csv', 'a+')
# csv_file = open('../cut_data/labels.csv', 'a+', newline='')
for image_name in data_set:
    # print(datum)
    img_b = cv2.imread(''.join((data_set_path, image_name[:2], '/', image_name, b, end)))
    img_b = np.array(img_b[:, :, 0], np.uint8)
    img_c = cv2.imread(''.join((data_set_path, image_name[:2], '/', image_name, c, end)))
    img_c = np.array(img_c[:, :, 0], np.uint8)

    img_b = pre_processing.adjust_average(img_b, img_c)
    img_b = pre_processing.cut_too_small(img_b, lambda_=1.4)
    img_b = pre_processing.cut_too_large(img_b)
    img_b = pre_processing.middle_filter(img_b)
    img_c = pre_processing.cut_too_small(img_c, lambda_=1.4)
    img_c = pre_processing.cut_too_large(img_c)
    img_c = pre_processing.middle_filter(img_c)

    cuts = pre_processing.cut(img_b, img_c)

    # break

csv_file.close()

model1 = classification.find_newest_model(model_path=model_path)

