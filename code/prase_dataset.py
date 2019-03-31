# coding=utf-8
import cv2
import numpy as np
import os
import csv
import matplotlib.pyplot as plt

train_image_path = '../af2019-cv-training-20190312/'
train_label_path = '../af2019-cv-training-20190312/list.csv'
index = 2458

train_data = []
with open(train_label_path) as csvfile:
    csv_reader = csv.reader(csvfile)
    data_header = next(csv_reader)
    for row in csv_reader:
        train_data.append(row)
print('The length of train_data is {}'.format(len(train_data)))
print('The last row of trainning data is ({})'.format(train_data[-1]))
info = train_data[index]
name = info[0]
x, y = int(info[1]), int(info[2])
class_1 = info[3]
# print(a[0])

shape_list = []
#
# # 尝试从直方图分析，no luck
#
# count = 0
# x_data = []
# y_data = []
# for data in train_data:
#     name = data[0]
#     x_data.append(data[1])
#     y_data.append(data[2])
# plt.hist(x_data, density=1, bins=40)
# plt.hist(y_data, density=1, bins=40)
#
# plt.show()

for data in train_data:
    if data[0] == '00b4b8a8152a05872297ec33fecac289':
        x = int(data[1])
        y = int(data[2])
        img_path = os.path.join(train_image_path, '00', '00aed3c6b8f351e52ed5075603b56be1_a.jpg')
        print(img_path)
        img = cv2.imread(img_path)
        # cv2.imshow('img', img)

        img = cv2.circle(img, (x, y), 5, (0, 255, 0), 1)
        cv2.imshow('img', img)
        cv2.waitKey(0)
# image_file_list = os.listdir(train_image_path)
# for idx in image_file_list:
#     if '.csv' not in idx and '.md' not in idx:
#         image_file = os.path.join(train_image_path, idx)
#         image_list = os.listdir(image_file)
#         for image in image_list:
#             if name in image:
#                 image_path = os.path.join(image_file, image)
#                 img = cv2.imread(image_path)
#                 img_shape = np.shape(img)
#                 # if(img_shape not ):
#                 # img = cv2.circle(img,(x,y),5,(0,255,0),1)
#                 # print(class_1)
#                 # cv2.imshow('img',img)
#                 # cv2.waitKey(0)
#             # if(os.path.exists(image_path)):
#             #     count +=1
# # print('We found %d image'%count)
