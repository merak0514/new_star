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
import resnet
import cluster
os.environ['CUDA_VISIBLE_DEVICES'] = '8'

b = '_b'
c = '_c'
end = '.jpg'

data_set_list_path = '../testb/list.csv'
data_set_path = '../testb/'
model_path = './classification/model50/'
data_set = []
with open(data_set_list_path) as f:
    csv_reader = csv.reader(f)
    for row in csv_reader:
        data_set.append(row[0])
    f.close()

print('The length of data is {}'.format(len(data_set)))

if input('Print y to continue') is not 'y':
    exit()

csv_file = open('./submit.csv', 'a+')
csv_write = csv.writer(csv_file, delimiter=',')
csv_write.writerow(['id', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'havestar'])
# csv_file = open('../cut_data/labels.csv', 'a+', newline='')
for image_name in data_set:
    # print(datum)
    img_b = cv2.imread(''.join((data_set_path, image_name[:2], '/', image_name, b, end)))
    img_b = np.array(img_b[:, :, 0], np.uint8)
    img_c = cv2.imread(''.join((data_set_path, image_name[:2], '/', image_name, c, end)))
    img_c = np.array(img_c[:, :, 0], np.uint8)

    img_b, img_c, black_count = pre_processing.cut_black(img_b, img_c, (0, 0), get_count=True)
    img_b = pre_processing.adjust_average(img_b, img_c)
    img_b = pre_processing.cut_too_small(img_b, lambda_=1.4)
    img_b = pre_processing.cut_too_large(img_b)
    img_b = pre_processing.middle_filter(img_b)
    img_c = pre_processing.cut_too_small(img_c, lambda_=1.4)
    img_c = pre_processing.cut_too_large(img_c)
    img_c = pre_processing.middle_filter(img_c)

    cuts_b, cuts_c, cut_pos = pre_processing.cut(img_b, img_c, (50, 50))

    resnet18 = resnet.resnet18().cuda()
    resnet18.eval()
    model = classification.find_newest_model(model_path_=model_path)

    imgs_b = torch.Tensor(cuts_b).cuda()
    imgs_c = torch.Tensor(cuts_c).cuda()
    image_combine = torch.cat((imgs_b.unsqueeze(3), imgs_c.unsqueeze(3)), dim=3).cuda()  # 作为二通道的输入
    image_combine = image_combine.permute((0, 3, 1, 2))
    outputs = resnet18.forward(image_combine)

    result = np.argmax(outputs[:, 2])
    coord = np.array(cut_pos[result]) + np.array(black_count[:2])
    b_result = cuts_b[result]
    c_result = cuts_c[result]

    result_image = pre_processing.compute_diff(b_result, c_result)

    # the function to give 3 pos
    answer = cluster.k_means(result_image, coord=coord)
    # end
    result_image = cv2.circle(ima_cut, (x, y), 15, 255, 1)

    result_image = cv2.rectangle(result_image, (x-30, y-30), (x+30, y+30), 255, 1)
    print(np.shape(imb))
    b_result = cv2.circle(b_result, (x, y), 15, 255, 1)
    c_result = cv2.circle(c_result, (x, y), 15, 255, 1)
    cv2.imshow('img_a', result_image)
    cv2.imshow('img_b', b_result)
    cv2.imshow('img_c', c_result)
    cv2.waitKey(0)


    row = [image_name, answer[0, 0], answer[1, 0], answer[0, 1], answer[1, 1], answer[0, 2], answer[1, 2], 0]
    # csv_write.writerow(row)

    break

csv_file.close()

model1 = classification.find_newest_model(model_path_=model_path)
